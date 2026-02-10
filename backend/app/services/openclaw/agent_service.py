"""Async agent lifecycle management service."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal, Protocol
from uuid import UUID, uuid4

from fastapi import HTTPException, Request, status
from sqlalchemy import asc, or_
from sqlmodel import col, select
from sse_starlette.sse import EventSourceResponse

from app.core.agent_tokens import generate_agent_token, hash_agent_token
from app.core.time import utcnow
from app.db import crud
from app.db.pagination import paginate
from app.db.session import async_session_maker
from app.integrations.openclaw_gateway import GatewayConfig as GatewayClientConfig
from app.integrations.openclaw_gateway import OpenClawGatewayError, ensure_session, send_message
from app.models.activity_events import ActivityEvent
from app.models.agents import Agent
from app.models.boards import Board
from app.models.gateways import Gateway
from app.models.organizations import Organization
from app.models.tasks import Task
from app.schemas.agents import (
    AgentCreate,
    AgentHeartbeat,
    AgentHeartbeatCreate,
    AgentRead,
    AgentUpdate,
)
from app.schemas.common import OkResponse
from app.services.activity_log import record_activity
from app.services.openclaw.constants import (
    AGENT_SESSION_PREFIX,
    DEFAULT_HEARTBEAT_CONFIG,
    OFFLINE_AFTER,
)
from app.services.openclaw.policies import OpenClawAuthorizationPolicy
from app.services.openclaw.provisioning import (
    AgentProvisionRequest,
    MainAgentProvisionRequest,
    ProvisionOptions,
    cleanup_agent,
    provision_agent,
    provision_main_agent,
)
from app.services.openclaw.shared import GatewayAgentIdentity
from app.services.organizations import (
    OrganizationContext,
    get_active_membership,
    has_board_access,
    is_org_admin,
    list_accessible_board_ids,
    require_board_access,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from fastapi_pagination.limit_offset import LimitOffsetPage
    from sqlalchemy.sql.elements import ColumnElement
    from sqlmodel.ext.asyncio.session import AsyncSession
    from sqlmodel.sql.expression import SelectOfScalar

    from app.models.users import User


class ActorContextLike(Protocol):
    """Minimal actor context contract consumed by lifecycle APIs."""

    actor_type: Literal["user", "agent"]
    user: User | None
    agent: Agent | None


@dataclass(frozen=True, slots=True)
class AgentUpdateOptions:
    """Runtime options for update-and-reprovision flows."""

    force: bool
    user: User | None
    context: OrganizationContext


@dataclass(frozen=True, slots=True)
class AgentUpdateProvisionTarget:
    """Resolved target for an update provision operation."""

    is_main_agent: bool
    board: Board | None
    gateway: Gateway
    client_config: GatewayClientConfig


@dataclass(frozen=True, slots=True)
class AgentUpdateProvisionRequest:
    """Provision request payload for agent updates."""

    target: AgentUpdateProvisionTarget
    raw_token: str
    user: User | None
    force_bootstrap: bool


class AbstractProvisionExecution(ABC):
    """Shared async execution contract for board/main agent provisioning actions."""

    def __init__(
        self,
        *,
        service: AgentLifecycleService,
        agent: Agent,
        provision_request: AgentUpdateProvisionRequest,
        action: str,
        wakeup_verb: str,
        raise_gateway_errors: bool,
    ) -> None:
        self._service = service
        self._agent = agent
        self._request = provision_request
        self._action = action
        self._wakeup_verb = wakeup_verb
        self._raise_gateway_errors = raise_gateway_errors

    @property
    def agent(self) -> Agent:
        return self._agent

    @agent.setter
    def agent(self, value: Agent) -> None:
        if not isinstance(value, Agent):
            msg = "agent must be an Agent model"
            raise TypeError(msg)
        self._agent = value

    @property
    def request(self) -> AgentUpdateProvisionRequest:
        return self._request

    @request.setter
    def request(self, value: AgentUpdateProvisionRequest) -> None:
        if not isinstance(value, AgentUpdateProvisionRequest):
            msg = "request must be an AgentUpdateProvisionRequest"
            raise TypeError(msg)
        self._request = value

    @property
    def logger(self) -> logging.Logger:
        return self._service.logger

    @abstractmethod
    async def _provision(self) -> None:
        raise NotImplementedError

    async def execute(self) -> None:
        self.logger.log(
            5,
            "agent.provision.start action=%s agent_id=%s target_main=%s",
            self._action,
            self.agent.id,
            self.request.target.is_main_agent,
        )
        try:
            await self._provision()
            await self._service.send_wakeup_message(
                self.agent,
                self.request.target.client_config,
                verb=self._wakeup_verb,
            )
            self.agent.provision_confirm_token_hash = None
            self.agent.provision_requested_at = None
            self.agent.provision_action = None
            self.agent.status = "online"
            self.agent.updated_at = utcnow()
            self._service.session.add(self.agent)
            await self._service.session.commit()
            record_activity(
                self._service.session,
                event_type=f"agent.{self._action}.direct",
                message=f"{self._action.capitalize()}d directly for {self.agent.name}.",
                agent_id=self.agent.id,
            )
            record_activity(
                self._service.session,
                event_type="agent.wakeup.sent",
                message=f"Wakeup message sent to {self.agent.name}.",
                agent_id=self.agent.id,
            )
            await self._service.session.commit()
            self.logger.info(
                "agent.provision.success action=%s agent_id=%s",
                self._action,
                self.agent.id,
            )
        except OpenClawGatewayError as exc:
            self._service.record_instruction_failure(
                self._service.session,
                self.agent,
                str(exc),
                self._action,
            )
            await self._service.session.commit()
            self.logger.error(
                "agent.provision.gateway_error action=%s agent_id=%s error=%s",
                self._action,
                self.agent.id,
                str(exc),
            )
            if self._raise_gateway_errors:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Gateway {self._action} failed: {exc}",
                ) from exc
        except (OSError, RuntimeError, ValueError) as exc:  # pragma: no cover
            self._service.record_instruction_failure(
                self._service.session,
                self.agent,
                str(exc),
                self._action,
            )
            await self._service.session.commit()
            self.logger.critical(
                "agent.provision.runtime_error action=%s agent_id=%s error=%s",
                self._action,
                self.agent.id,
                str(exc),
            )
            if self._raise_gateway_errors:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Unexpected error {self._action}ing agent provisioning.",
                ) from exc


class BoardAgentProvisionExecution(AbstractProvisionExecution):
    """Provision execution for board-scoped agents."""

    async def _provision(self) -> None:
        board = self.request.target.board
        if board is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="board is required for non-main agent provisioning",
            )
        await provision_agent(
            self.agent,
            AgentProvisionRequest(
                board=board,
                gateway=self.request.target.gateway,
                auth_token=self.request.raw_token,
                user=self.request.user,
                options=ProvisionOptions(
                    action=self._action,
                    force_bootstrap=self.request.force_bootstrap,
                    reset_session=True,
                ),
            ),
        )


class MainAgentProvisionExecution(AbstractProvisionExecution):
    """Provision execution for gateway-main agents."""

    async def _provision(self) -> None:
        await provision_main_agent(
            self.agent,
            MainAgentProvisionRequest(
                gateway=self.request.target.gateway,
                auth_token=self.request.raw_token,
                user=self.request.user,
                session_key=self.agent.openclaw_session_id,
                options=ProvisionOptions(
                    action=self._action,
                    force_bootstrap=self.request.force_bootstrap,
                    reset_session=True,
                ),
            ),
        )


class AgentLifecycleService:
    """Async service encapsulating agent lifecycle behavior for API routes."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._logger = logging.getLogger(__name__)

    @property
    def session(self) -> AsyncSession:
        return self._session

    @session.setter
    def session(self, value: AsyncSession) -> None:
        self._session = value

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    @logger.setter
    def logger(self, value: logging.Logger) -> None:
        self._logger = value

    @staticmethod
    def parse_since(value: str | None) -> datetime | None:
        if not value:
            return None
        normalized = value.strip()
        if not normalized:
            return None
        normalized = normalized.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
        if parsed.tzinfo is not None:
            return parsed.astimezone(UTC).replace(tzinfo=None)
        return parsed

    @staticmethod
    def slugify(value: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
        return slug or uuid4().hex

    @classmethod
    def build_session_key(cls, agent_name: str) -> str:
        return f"{AGENT_SESSION_PREFIX}:{cls.slugify(agent_name)}:main"

    @classmethod
    def workspace_path(cls, agent_name: str, workspace_root: str | None) -> str:
        if not workspace_root:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Gateway workspace_root is required",
            )
        root = workspace_root.rstrip("/")
        return f"{root}/workspace-{cls.slugify(agent_name)}"

    async def require_board(
        self,
        board_id: UUID | str | None,
        *,
        user: User | None = None,
        write: bool = False,
    ) -> Board:
        if not board_id:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="board_id is required",
            )
        board = await Board.objects.by_id(board_id).first(self.session)
        if board is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Board not found",
            )
        if user is not None:
            await require_board_access(self.session, user=user, board=board, write=write)
        return board

    async def require_gateway(
        self,
        board: Board,
    ) -> tuple[Gateway, GatewayClientConfig]:
        if not board.gateway_id:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Board gateway_id is required",
            )
        gateway = await Gateway.objects.by_id(board.gateway_id).first(self.session)
        if gateway is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Board gateway_id is invalid",
            )
        if gateway.organization_id != board.organization_id:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Board gateway_id is invalid",
            )
        if not gateway.url:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Gateway url is required",
            )
        if not gateway.workspace_root:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Gateway workspace_root is required",
            )
        return gateway, GatewayClientConfig(url=gateway.url, token=gateway.token)

    @staticmethod
    def gateway_client_config(gateway: Gateway) -> GatewayClientConfig:
        if not gateway.url:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Gateway url is required",
            )
        return GatewayClientConfig(url=gateway.url, token=gateway.token)

    @staticmethod
    def is_gateway_main(agent: Agent) -> bool:
        return agent.board_id is None

    @classmethod
    def to_agent_read(cls, agent: Agent) -> AgentRead:
        model = AgentRead.model_validate(agent, from_attributes=True)
        return model.model_copy(
            update={"is_gateway_main": cls.is_gateway_main(agent)},
        )

    @staticmethod
    def coerce_agent_items(items: Sequence[Any]) -> list[Agent]:
        agents: list[Agent] = []
        for item in items:
            if not isinstance(item, Agent):
                msg = "Expected Agent items from paginated query"
                raise TypeError(msg)
            agents.append(item)
        return agents

    async def get_main_agent_gateway(self, agent: Agent) -> Gateway | None:
        if agent.board_id is not None:
            return None
        return await Gateway.objects.by_id(agent.gateway_id).first(self.session)

    async def ensure_gateway_session(
        self,
        agent_name: str,
        config: GatewayClientConfig,
    ) -> tuple[str, str | None]:
        session_key = self.build_session_key(agent_name)
        try:
            await ensure_session(session_key, config=config, label=agent_name)
        except OpenClawGatewayError as exc:
            self.logger.warning(
                "agent.session.ensure_failed agent_name=%s error=%s",
                agent_name,
                str(exc),
            )
            return session_key, str(exc)
        return session_key, None

    @classmethod
    def with_computed_status(cls, agent: Agent) -> Agent:
        now = utcnow()
        if agent.status in {"deleting", "updating"}:
            return agent
        if agent.last_seen_at is None:
            agent.status = "provisioning"
        elif now - agent.last_seen_at > OFFLINE_AFTER:
            agent.status = "offline"
        return agent

    @classmethod
    def serialize_agent(cls, agent: Agent) -> dict[str, object]:
        return cls.to_agent_read(cls.with_computed_status(agent)).model_dump(mode="json")

    async def fetch_agent_events(
        self,
        board_id: UUID | None,
        since: datetime,
    ) -> list[Agent]:
        statement = select(Agent)
        if board_id:
            statement = statement.where(col(Agent.board_id) == board_id)
        statement = statement.where(
            or_(
                col(Agent.updated_at) >= since,
                col(Agent.last_seen_at) >= since,
            ),
        ).order_by(asc(col(Agent.updated_at)))
        return list(await self.session.exec(statement))

    async def require_user_context(self, user: User | None) -> OrganizationContext:
        if user is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
        member = await get_active_membership(self.session, user)
        if member is None:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
        organization = await Organization.objects.by_id(member.organization_id).first(self.session)
        if organization is None:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
        return OrganizationContext(organization=organization, member=member)

    async def require_agent_access(
        self,
        *,
        agent: Agent,
        ctx: OrganizationContext,
        write: bool,
    ) -> None:
        if agent.board_id is None:
            OpenClawAuthorizationPolicy.require_org_admin(is_admin=is_org_admin(ctx.member))
            gateway = await self.get_main_agent_gateway(agent)
            OpenClawAuthorizationPolicy.require_gateway_in_org(
                gateway=gateway,
                organization_id=ctx.organization.id,
            )
            return

        board = await Board.objects.by_id(agent.board_id).first(self.session)
        board = OpenClawAuthorizationPolicy.require_board_in_org(
            board=board,
            organization_id=ctx.organization.id,
        )
        allowed = await has_board_access(
            self.session,
            member=ctx.member,
            board=board,
            write=write,
        )
        OpenClawAuthorizationPolicy.require_board_write_access(allowed=allowed)

    @staticmethod
    def record_heartbeat(session: AsyncSession, agent: Agent) -> None:
        record_activity(
            session,
            event_type="agent.heartbeat",
            message=f"Heartbeat received from {agent.name}.",
            agent_id=agent.id,
        )

    @staticmethod
    def record_instruction_failure(
        session: AsyncSession,
        agent: Agent,
        error: str,
        action: str,
    ) -> None:
        action_label = action.replace("_", " ").capitalize()
        record_activity(
            session,
            event_type=f"agent.{action}.failed",
            message=f"{action_label} message failed: {error}",
            agent_id=agent.id,
        )

    async def coerce_agent_create_payload(
        self,
        payload: AgentCreate,
        actor: ActorContextLike,
    ) -> AgentCreate:
        if actor.actor_type == "user":
            ctx = await self.require_user_context(actor.user)
            OpenClawAuthorizationPolicy.require_org_admin(is_admin=is_org_admin(ctx.member))
            return payload

        if actor.actor_type == "agent":
            board_id = OpenClawAuthorizationPolicy.resolve_board_lead_create_board_id(
                actor_agent=actor.agent,
                requested_board_id=payload.board_id,
            )
            return AgentCreate(**{**payload.model_dump(), "board_id": board_id})

        return payload

    async def ensure_unique_agent_name(
        self,
        *,
        board: Board,
        gateway: Gateway,
        requested_name: str,
    ) -> None:
        if not requested_name:
            return

        existing = (
            await self.session.exec(
                select(Agent)
                .where(Agent.board_id == board.id)
                .where(col(Agent.name).ilike(requested_name)),
            )
        ).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="An agent with this name already exists on this board.",
            )

        existing_gateway = (
            await self.session.exec(
                select(Agent)
                .join(Board, col(Agent.board_id) == col(Board.id))
                .where(col(Board.gateway_id) == gateway.id)
                .where(col(Agent.name).ilike(requested_name)),
            )
        ).first()
        if existing_gateway:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="An agent with this name already exists in this gateway workspace.",
            )

        desired_session_key = self.build_session_key(requested_name)
        existing_session_key = (
            await self.session.exec(
                select(Agent)
                .join(Board, col(Agent.board_id) == col(Board.id))
                .where(col(Board.gateway_id) == gateway.id)
                .where(col(Agent.openclaw_session_id) == desired_session_key),
            )
        ).first()
        if existing_session_key:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "This agent name would collide with an existing workspace "
                    "session key. Pick a different name."
                ),
            )

    async def persist_new_agent(
        self,
        *,
        data: dict[str, Any],
        client_config: GatewayClientConfig,
    ) -> tuple[Agent, str, str | None]:
        agent = Agent.model_validate(data)
        agent.status = "provisioning"
        raw_token = generate_agent_token()
        agent.agent_token_hash = hash_agent_token(raw_token)
        if agent.heartbeat_config is None:
            agent.heartbeat_config = DEFAULT_HEARTBEAT_CONFIG.copy()
        agent.provision_requested_at = utcnow()
        agent.provision_action = "provision"
        session_key, session_error = await self.ensure_gateway_session(
            agent.name,
            client_config,
        )
        agent.openclaw_session_id = session_key
        self.session.add(agent)
        await self.session.commit()
        await self.session.refresh(agent)
        return agent, raw_token, session_error

    async def record_session_creation(
        self,
        *,
        agent: Agent,
        session_error: str | None,
    ) -> None:
        if session_error:
            record_activity(
                self.session,
                event_type="agent.session.failed",
                message=f"Session sync failed for {agent.name}: {session_error}",
                agent_id=agent.id,
            )
        else:
            record_activity(
                self.session,
                event_type="agent.session.created",
                message=f"Session created for {agent.name}.",
                agent_id=agent.id,
            )
        await self.session.commit()

    async def send_wakeup_message(
        self,
        agent: Agent,
        config: GatewayClientConfig,
        verb: str = "provisioned",
    ) -> None:
        session_key = agent.openclaw_session_id or self.build_session_key(agent.name)
        await ensure_session(session_key, config=config, label=agent.name)
        message = (
            f"Hello {agent.name}. Your workspace has been {verb}.\n\n"
            "Start the agent, run BOOT.md, and if BOOTSTRAP.md exists run it once "
            "then delete it. Begin heartbeats after startup."
        )
        await send_message(message, session_key=session_key, config=config, deliver=True)

    async def provision_new_agent(
        self,
        *,
        agent: Agent,
        request: AgentProvisionRequest,
        client_config: GatewayClientConfig,
    ) -> None:
        execution = BoardAgentProvisionExecution(
            service=self,
            agent=agent,
            provision_request=AgentUpdateProvisionRequest(
                target=AgentUpdateProvisionTarget(
                    is_main_agent=False,
                    board=request.board,
                    gateway=request.gateway,
                    client_config=client_config,
                ),
                raw_token=request.auth_token,
                user=request.user,
                force_bootstrap=request.options.force_bootstrap,
            ),
            action="provision",
            wakeup_verb="provisioned",
            raise_gateway_errors=False,
        )
        await execution.execute()

    async def validate_agent_update_inputs(
        self,
        *,
        ctx: OrganizationContext,
        updates: dict[str, Any],
        make_main: bool | None,
    ) -> None:
        if make_main:
            OpenClawAuthorizationPolicy.require_org_admin(is_admin=is_org_admin(ctx.member))
        if "status" in updates:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="status is controlled by agent heartbeat",
            )
        if "board_id" in updates and updates["board_id"] is not None:
            new_board = await self.require_board(updates["board_id"])
            OpenClawAuthorizationPolicy.require_board_in_org(
                board=new_board,
                organization_id=ctx.organization.id,
            )
            allowed = await has_board_access(
                self.session,
                member=ctx.member,
                board=new_board,
                write=True,
            )
            OpenClawAuthorizationPolicy.require_board_write_access(allowed=allowed)

    async def apply_agent_update_mutations(
        self,
        *,
        agent: Agent,
        updates: dict[str, Any],
        make_main: bool | None,
    ) -> tuple[Gateway | None, Gateway | None]:
        main_gateway = await self.get_main_agent_gateway(agent)
        gateway_for_main: Gateway | None = None

        if make_main:
            board_source = updates.get("board_id") or agent.board_id
            board_for_main = await self.require_board(board_source)
            gateway_for_main, _ = await self.require_gateway(board_for_main)
            updates["board_id"] = None
            updates["gateway_id"] = gateway_for_main.id
            agent.is_board_lead = False
            agent.openclaw_session_id = GatewayAgentIdentity.session_key(gateway_for_main)
            main_gateway = gateway_for_main
        elif make_main is not None:
            if "board_id" not in updates or updates["board_id"] is None:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=(
                        "board_id is required when converting a gateway-main agent "
                        "to board scope"
                    ),
                )
            board = await self.require_board(updates["board_id"])
            if board.gateway_id is None:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Board gateway_id is required",
                )
            updates["gateway_id"] = board.gateway_id
            agent.openclaw_session_id = None

        if make_main is None and "board_id" in updates:
            board = await self.require_board(updates["board_id"])
            if board.gateway_id is None:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Board gateway_id is required",
                )
            updates["gateway_id"] = board.gateway_id
        for key, value in updates.items():
            setattr(agent, key, value)

        if make_main is None and main_gateway is not None:
            agent.board_id = None
            agent.gateway_id = main_gateway.id
            agent.is_board_lead = False
        if make_main is False and agent.board_id is not None:
            board = await self.require_board(agent.board_id)
            if board.gateway_id is None:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Board gateway_id is required",
                )
            agent.gateway_id = board.gateway_id
        agent.updated_at = utcnow()
        if agent.heartbeat_config is None:
            agent.heartbeat_config = DEFAULT_HEARTBEAT_CONFIG.copy()
        self.session.add(agent)
        await self.session.commit()
        await self.session.refresh(agent)
        return main_gateway, gateway_for_main

    async def resolve_agent_update_target(
        self,
        *,
        agent: Agent,
        make_main: bool | None,
        main_gateway: Gateway | None,
        gateway_for_main: Gateway | None,
    ) -> AgentUpdateProvisionTarget:
        if make_main:
            if gateway_for_main is None:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Gateway agent requires a gateway configuration",
                )
            return AgentUpdateProvisionTarget(
                is_main_agent=True,
                board=None,
                gateway=gateway_for_main,
                client_config=self.gateway_client_config(gateway_for_main),
            )

        if make_main is None and agent.board_id is None and main_gateway is not None:
            return AgentUpdateProvisionTarget(
                is_main_agent=True,
                board=None,
                gateway=main_gateway,
                client_config=self.gateway_client_config(main_gateway),
            )

        if agent.board_id is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="board_id is required for non-main agents",
            )
        board = await self.require_board(agent.board_id)
        gateway, client_config = await self.require_gateway(board)
        return AgentUpdateProvisionTarget(
            is_main_agent=False,
            board=board,
            gateway=gateway,
            client_config=client_config,
        )

    async def ensure_agent_update_session(
        self,
        *,
        agent: Agent,
        client_config: GatewayClientConfig,
    ) -> None:
        session_key = agent.openclaw_session_id or self.build_session_key(agent.name)
        try:
            await ensure_session(session_key, config=client_config, label=agent.name)
            if not agent.openclaw_session_id:
                agent.openclaw_session_id = session_key
                self.session.add(agent)
                await self.session.commit()
                await self.session.refresh(agent)
        except OpenClawGatewayError as exc:
            self.record_instruction_failure(self.session, agent, str(exc), "update")
            await self.session.commit()

    @staticmethod
    def mark_agent_update_pending(agent: Agent) -> str:
        raw_token = generate_agent_token()
        agent.agent_token_hash = hash_agent_token(raw_token)
        agent.provision_requested_at = utcnow()
        agent.provision_action = "update"
        agent.status = "updating"
        return raw_token

    async def provision_updated_agent(
        self,
        *,
        agent: Agent,
        request: AgentUpdateProvisionRequest,
    ) -> None:
        execution: AbstractProvisionExecution
        if request.target.is_main_agent:
            execution = MainAgentProvisionExecution(
                service=self,
                agent=agent,
                provision_request=request,
                action="update",
                wakeup_verb="updated",
                raise_gateway_errors=True,
            )
        else:
            execution = BoardAgentProvisionExecution(
                service=self,
                agent=agent,
                provision_request=request,
                action="update",
                wakeup_verb="updated",
                raise_gateway_errors=True,
            )
        await execution.execute()

    @staticmethod
    def heartbeat_lookup_statement(payload: AgentHeartbeatCreate) -> SelectOfScalar[Agent]:
        statement = Agent.objects.filter_by(name=payload.name).statement
        if payload.board_id is not None:
            statement = statement.where(Agent.board_id == payload.board_id)
        return statement

    async def create_agent_from_heartbeat(
        self,
        *,
        payload: AgentHeartbeatCreate,
        actor: ActorContextLike,
    ) -> Agent:
        if actor.actor_type == "agent":
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
        if actor.actor_type == "user":
            ctx = await self.require_user_context(actor.user)
            OpenClawAuthorizationPolicy.require_org_admin(is_admin=is_org_admin(ctx.member))

        board = await self.require_board(
            payload.board_id,
            user=actor.user,
            write=True,
        )
        gateway, client_config = await self.require_gateway(board)
        data: dict[str, Any] = {
            "name": payload.name,
            "board_id": board.id,
            "gateway_id": gateway.id,
            "heartbeat_config": DEFAULT_HEARTBEAT_CONFIG.copy(),
        }
        agent, raw_token, session_error = await self.persist_new_agent(
            data=data,
            client_config=client_config,
        )
        await self.record_session_creation(
            agent=agent,
            session_error=session_error,
        )
        await self.provision_new_agent(
            agent=agent,
            request=AgentProvisionRequest(
                board=board,
                gateway=gateway,
                auth_token=raw_token,
                user=actor.user,
                options=ProvisionOptions(action="provision"),
            ),
            client_config=client_config,
        )
        return agent

    async def handle_existing_user_heartbeat_agent(
        self,
        *,
        agent: Agent,
        user: User | None,
    ) -> None:
        ctx = await self.require_user_context(user)
        await self.require_agent_access(agent=agent, ctx=ctx, write=True)

        if agent.agent_token_hash is not None:
            return

        raw_token = generate_agent_token()
        agent.agent_token_hash = hash_agent_token(raw_token)
        if agent.heartbeat_config is None:
            agent.heartbeat_config = DEFAULT_HEARTBEAT_CONFIG.copy()
        agent.provision_requested_at = utcnow()
        agent.provision_action = "provision"
        self.session.add(agent)
        await self.session.commit()
        await self.session.refresh(agent)
        board = await self.require_board(
            str(agent.board_id) if agent.board_id else None,
            user=user,
            write=True,
        )
        gateway, client_config = await self.require_gateway(board)
        await self.provision_new_agent(
            agent=agent,
            request=AgentProvisionRequest(
                board=board,
                gateway=gateway,
                auth_token=raw_token,
                user=user,
                options=ProvisionOptions(action="provision"),
            ),
            client_config=client_config,
        )

    async def ensure_heartbeat_session_key(
        self,
        *,
        agent: Agent,
        actor: ActorContextLike,
    ) -> None:
        if agent.openclaw_session_id:
            return
        board = await self.require_board(
            str(agent.board_id) if agent.board_id else None,
            user=actor.user if actor.actor_type == "user" else None,
            write=actor.actor_type == "user",
        )
        _, client_config = await self.require_gateway(board)
        session_key, session_error = await self.ensure_gateway_session(
            agent.name,
            client_config,
        )
        agent.openclaw_session_id = session_key
        self.session.add(agent)
        await self.record_session_creation(
            agent=agent,
            session_error=session_error,
        )

    async def commit_heartbeat(
        self,
        *,
        agent: Agent,
        status_value: str | None,
    ) -> AgentRead:
        if status_value:
            agent.status = status_value
        elif agent.status == "provisioning":
            agent.status = "online"
        agent.last_seen_at = utcnow()
        agent.updated_at = utcnow()
        self.record_heartbeat(self.session, agent)
        self.session.add(agent)
        await self.session.commit()
        await self.session.refresh(agent)
        return self.to_agent_read(self.with_computed_status(agent))

    async def list_agents(
        self,
        *,
        board_id: UUID | None,
        gateway_id: UUID | None,
        ctx: OrganizationContext,
    ) -> LimitOffsetPage[AgentRead]:
        board_ids = await list_accessible_board_ids(self.session, member=ctx.member, write=False)
        if board_id is not None:
            OpenClawAuthorizationPolicy.require_board_write_access(
                allowed=board_id in set(board_ids),
            )
        base_filters: list[ColumnElement[bool]] = []
        if board_ids:
            base_filters.append(col(Agent.board_id).in_(board_ids))
        if is_org_admin(ctx.member):
            gateways = await Gateway.objects.filter_by(
                organization_id=ctx.organization.id,
            ).all(self.session)
            gateway_ids = [gateway.id for gateway in gateways]
            if gateway_ids:
                base_filters.append(
                    (col(Agent.gateway_id).in_(gateway_ids)) & (col(Agent.board_id).is_(None)),
                )
        if base_filters:
            if len(base_filters) == 1:
                statement = select(Agent).where(base_filters[0])
            else:
                statement = select(Agent).where(or_(*base_filters))
        else:
            statement = select(Agent).where(col(Agent.id).is_(None))
        if board_id is not None:
            statement = statement.where(col(Agent.board_id) == board_id)
        if gateway_id is not None:
            gateway = await Gateway.objects.by_id(gateway_id).first(self.session)
            if gateway is None or gateway.organization_id != ctx.organization.id:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
            gateway_board_ids = select(Board.id).where(col(Board.gateway_id) == gateway_id)
            statement = statement.where(
                or_(
                    col(Agent.board_id).in_(gateway_board_ids),
                    (col(Agent.gateway_id) == gateway_id) & (col(Agent.board_id).is_(None)),
                ),
            )
        statement = statement.order_by(col(Agent.created_at).desc())

        def _transform(items: Sequence[Any]) -> Sequence[Any]:
            agents = self.coerce_agent_items(items)
            return [self.to_agent_read(self.with_computed_status(agent)) for agent in agents]

        return await paginate(self.session, statement, transformer=_transform)

    async def stream_agents(
        self,
        *,
        request: Request,
        board_id: UUID | None,
        since: str | None,
        ctx: OrganizationContext,
    ) -> EventSourceResponse:
        since_dt = self.parse_since(since) or utcnow()
        last_seen = since_dt
        board_ids = await list_accessible_board_ids(self.session, member=ctx.member, write=False)
        allowed_ids = set(board_ids)
        if board_id is not None:
            OpenClawAuthorizationPolicy.require_board_write_access(allowed=board_id in allowed_ids)

        async def event_generator() -> AsyncIterator[dict[str, str]]:
            nonlocal last_seen
            while True:
                if await request.is_disconnected():
                    break
                async with async_session_maker() as stream_session:
                    stream_service = AgentLifecycleService(stream_session)
                    stream_service.logger = self.logger
                    if board_id is not None:
                        agents = await stream_service.fetch_agent_events(
                            board_id,
                            last_seen,
                        )
                    elif allowed_ids:
                        agents = await stream_service.fetch_agent_events(None, last_seen)
                        agents = [agent for agent in agents if agent.board_id in allowed_ids]
                    else:
                        agents = []
                for agent in agents:
                    updated_at = agent.updated_at or agent.last_seen_at or utcnow()
                    last_seen = max(updated_at, last_seen)
                    payload = {"agent": self.serialize_agent(agent)}
                    yield {"event": "agent", "data": json.dumps(payload)}
                await asyncio.sleep(2)

        return EventSourceResponse(event_generator(), ping=15)

    async def create_agent(
        self,
        *,
        payload: AgentCreate,
        actor: ActorContextLike,
    ) -> AgentRead:
        self.logger.log(
            5,
            "agent.create.start actor_type=%s board_id=%s",
            actor.actor_type,
            payload.board_id,
        )
        payload = await self.coerce_agent_create_payload(payload, actor)

        board = await self.require_board(
            payload.board_id,
            user=actor.user if actor.actor_type == "user" else None,
            write=actor.actor_type == "user",
        )
        gateway, client_config = await self.require_gateway(board)
        data = payload.model_dump()
        data["gateway_id"] = gateway.id
        requested_name = (data.get("name") or "").strip()
        await self.ensure_unique_agent_name(
            board=board,
            gateway=gateway,
            requested_name=requested_name,
        )
        agent, raw_token, session_error = await self.persist_new_agent(
            data=data,
            client_config=client_config,
        )
        await self.record_session_creation(
            agent=agent,
            session_error=session_error,
        )
        provision_request = AgentProvisionRequest(
            board=board,
            gateway=gateway,
            auth_token=raw_token,
            user=actor.user if actor.actor_type == "user" else None,
            options=ProvisionOptions(action="provision"),
        )
        await self.provision_new_agent(
            agent=agent,
            request=provision_request,
            client_config=client_config,
        )
        self.logger.info("agent.create.success agent_id=%s board_id=%s", agent.id, board.id)
        return self.to_agent_read(self.with_computed_status(agent))

    async def get_agent(
        self,
        *,
        agent_id: str,
        ctx: OrganizationContext,
    ) -> AgentRead:
        agent = await Agent.objects.by_id(agent_id).first(self.session)
        if agent is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
        await self.require_agent_access(agent=agent, ctx=ctx, write=False)
        return self.to_agent_read(self.with_computed_status(agent))

    async def update_agent(
        self,
        *,
        agent_id: str,
        payload: AgentUpdate,
        options: AgentUpdateOptions,
    ) -> AgentRead:
        self.logger.log(5, "agent.update.start agent_id=%s force=%s", agent_id, options.force)
        agent = await Agent.objects.by_id(agent_id).first(self.session)
        if agent is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
        await self.require_agent_access(agent=agent, ctx=options.context, write=True)
        updates = payload.model_dump(exclude_unset=True)
        make_main = updates.pop("is_gateway_main", None)
        await self.validate_agent_update_inputs(
            ctx=options.context,
            updates=updates,
            make_main=make_main,
        )
        if not updates and not options.force and make_main is None:
            return self.to_agent_read(self.with_computed_status(agent))
        main_gateway, gateway_for_main = await self.apply_agent_update_mutations(
            agent=agent,
            updates=updates,
            make_main=make_main,
        )
        target = await self.resolve_agent_update_target(
            agent=agent,
            make_main=make_main,
            main_gateway=main_gateway,
            gateway_for_main=gateway_for_main,
        )
        await self.ensure_agent_update_session(
            agent=agent,
            client_config=target.client_config,
        )
        raw_token = self.mark_agent_update_pending(agent)
        self.session.add(agent)
        await self.session.commit()
        await self.session.refresh(agent)
        provision_request = AgentUpdateProvisionRequest(
            target=target,
            raw_token=raw_token,
            user=options.user,
            force_bootstrap=options.force,
        )
        await self.provision_updated_agent(
            agent=agent,
            request=provision_request,
        )
        self.logger.info("agent.update.success agent_id=%s", agent.id)
        return self.to_agent_read(self.with_computed_status(agent))

    async def heartbeat_agent(
        self,
        *,
        agent_id: str,
        payload: AgentHeartbeat,
        actor: ActorContextLike,
    ) -> AgentRead:
        self.logger.log(
            5, "agent.heartbeat.start agent_id=%s actor_type=%s", agent_id, actor.actor_type
        )
        agent = await Agent.objects.by_id(agent_id).first(self.session)
        if agent is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
        if actor.actor_type == "agent":
            OpenClawAuthorizationPolicy.require_same_agent_actor(
                actor_agent_id=actor.agent.id if actor.agent else None,
                target_agent_id=agent.id,
            )
        if actor.actor_type == "user":
            ctx = await self.require_user_context(actor.user)
            OpenClawAuthorizationPolicy.require_org_admin(is_admin=is_org_admin(ctx.member))
            await self.require_agent_access(agent=agent, ctx=ctx, write=True)
        return await self.commit_heartbeat(
            agent=agent,
            status_value=payload.status,
        )

    async def heartbeat_or_create_agent(
        self,
        *,
        payload: AgentHeartbeatCreate,
        actor: ActorContextLike,
    ) -> AgentRead:
        self.logger.log(
            5,
            "agent.heartbeat_or_create.start actor_type=%s name=%s board_id=%s",
            actor.actor_type,
            payload.name,
            payload.board_id,
        )
        if actor.actor_type == "agent" and actor.agent:
            return await self.heartbeat_agent(
                agent_id=str(actor.agent.id),
                payload=AgentHeartbeat(status=payload.status),
                actor=actor,
            )

        agent = (await self.session.exec(self.heartbeat_lookup_statement(payload))).first()
        if agent is None:
            agent = await self.create_agent_from_heartbeat(
                payload=payload,
                actor=actor,
            )
        elif actor.actor_type == "user":
            await self.handle_existing_user_heartbeat_agent(
                agent=agent,
                user=actor.user,
            )
        elif actor.actor_type == "agent":
            OpenClawAuthorizationPolicy.require_same_agent_actor(
                actor_agent_id=actor.agent.id if actor.agent else None,
                target_agent_id=agent.id,
            )

        await self.ensure_heartbeat_session_key(
            agent=agent,
            actor=actor,
        )
        return await self.commit_heartbeat(
            agent=agent,
            status_value=payload.status,
        )

    async def delete_agent(
        self,
        *,
        agent_id: str,
        ctx: OrganizationContext,
    ) -> OkResponse:
        self.logger.log(5, "agent.delete.start agent_id=%s", agent_id)
        agent = await Agent.objects.by_id(agent_id).first(self.session)
        if agent is None:
            return OkResponse()
        await self.require_agent_access(agent=agent, ctx=ctx, write=True)

        board = await self.require_board(str(agent.board_id) if agent.board_id else None)
        gateway, client_config = await self.require_gateway(board)
        try:
            workspace_path = await cleanup_agent(agent, gateway)
        except OpenClawGatewayError as exc:
            self.record_instruction_failure(self.session, agent, str(exc), "delete")
            await self.session.commit()
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Gateway cleanup failed: {exc}",
            ) from exc
        except (OSError, RuntimeError, ValueError) as exc:  # pragma: no cover
            self.record_instruction_failure(self.session, agent, str(exc), "delete")
            await self.session.commit()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Workspace cleanup failed: {exc}",
            ) from exc

        record_activity(
            self.session,
            event_type="agent.delete.direct",
            message=f"Deleted agent {agent.name}.",
            agent_id=None,
        )
        now = utcnow()
        await crud.update_where(
            self.session,
            Task,
            col(Task.assigned_agent_id) == agent.id,
            col(Task.status) == "in_progress",
            assigned_agent_id=None,
            status="inbox",
            in_progress_at=None,
            updated_at=now,
            commit=False,
        )
        await crud.update_where(
            self.session,
            Task,
            col(Task.assigned_agent_id) == agent.id,
            col(Task.status) != "in_progress",
            assigned_agent_id=None,
            updated_at=now,
            commit=False,
        )
        await crud.update_where(
            self.session,
            ActivityEvent,
            col(ActivityEvent.agent_id) == agent.id,
            agent_id=None,
            commit=False,
        )
        await self.session.delete(agent)
        await self.session.commit()

        try:
            main_session = GatewayAgentIdentity.session_key(gateway)
            if main_session and workspace_path:
                cleanup_message = (
                    "Cleanup request for deleted agent.\n\n"
                    f"Agent name: {agent.name}\n"
                    f"Agent id: {agent.id}\n"
                    f"Workspace path: {workspace_path}\n\n"
                    "Actions:\n"
                    "1) Remove the workspace directory.\n"
                    "2) Reply NO_REPLY.\n"
                )
                await ensure_session(main_session, config=client_config, label="Gateway Agent")
                await send_message(
                    cleanup_message,
                    session_key=main_session,
                    config=client_config,
                    deliver=False,
                )
        except (OSError, OpenClawGatewayError, ValueError):
            pass
        self.logger.info("agent.delete.success agent_id=%s", agent_id)
        return OkResponse()
