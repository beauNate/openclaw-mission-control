"""Async agent lifecycle management service."""

from __future__ import annotations

import asyncio
import json
import logging
import re
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
from app.integrations.openclaw_gateway import (
    OpenClawGatewayError,
    ensure_session,
    send_message,
)
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
from app.services.openclaw.provisioning import OpenClawGatewayProvisioner
from app.services.openclaw.shared import GatewayAgentIdentity
from app.services.organizations import (
    OrganizationContext,
    get_active_membership,
    get_org_owner_user,
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


@dataclass(frozen=True, slots=True)
class AgentUpdateProvisionRequest:
    """Provision request payload for agent updates."""

    target: AgentUpdateProvisionTarget
    raw_token: str
    user: User | None
    force_bootstrap: bool


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
    def resolve_session_key(cls, agent: Agent) -> str:
        """Resolve the gateway session key for an agent.

        Notes:
        - For board-scoped agents, default to a UUID-based key to avoid name collisions.
        """

        existing = (agent.openclaw_session_id or "").strip()
        if agent.board_id is None:
            # Gateway-main agents must have an explicit deterministic key (set elsewhere).
            if existing:
                return existing
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Gateway main agent session key is required",
            )
        if agent.is_board_lead:
            return f"{AGENT_SESSION_PREFIX}:lead-{agent.board_id}:main"
        return f"{AGENT_SESSION_PREFIX}:mc-{agent.id}:main"

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

    async def persist_new_agent(
        self,
        *,
        data: dict[str, Any],
    ) -> tuple[Agent, str]:
        agent = Agent.model_validate(data)
        agent.status = "provisioning"
        raw_token = generate_agent_token()
        agent.agent_token_hash = hash_agent_token(raw_token)
        if agent.heartbeat_config is None:
            agent.heartbeat_config = DEFAULT_HEARTBEAT_CONFIG.copy()
        agent.provision_requested_at = utcnow()
        agent.provision_action = "provision"
        agent.openclaw_session_id = self.resolve_session_key(agent)
        self.session.add(agent)
        await self.session.commit()
        await self.session.refresh(agent)
        return agent, raw_token

    async def _apply_gateway_provisioning(
        self,
        *,
        agent: Agent,
        target: AgentUpdateProvisionTarget,
        auth_token: str,
        user: User | None,
        action: str,
        wakeup_verb: str,
        force_bootstrap: bool,
        raise_gateway_errors: bool,
    ) -> None:
        self.logger.log(
            5,
            "agent.provision.start action=%s agent_id=%s target_main=%s",
            action,
            agent.id,
            target.is_main_agent,
        )
        try:
            if not target.is_main_agent and target.board is None:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="board is required for non-main agent provisioning",
                )
            template_user = user
            if target.is_main_agent and template_user is None:
                template_user = await get_org_owner_user(
                    self.session,
                    organization_id=target.gateway.organization_id,
                )
                if template_user is None:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail=(
                            "User context is required to provision the gateway main agent "
                            "(org owner not found)."
                        ),
                    )
            await OpenClawGatewayProvisioner().apply_agent_lifecycle(
                agent=agent,
                gateway=target.gateway,
                board=target.board if not target.is_main_agent else None,
                auth_token=auth_token,
                user=template_user,
                action=action,
                force_bootstrap=force_bootstrap,
                reset_session=True,
                wake=True,
                deliver_wakeup=True,
                wakeup_verb=wakeup_verb,
            )
            agent.provision_confirm_token_hash = None
            agent.provision_requested_at = None
            agent.provision_action = None
            agent.status = "online"
            agent.updated_at = utcnow()
            self.session.add(agent)
            await self.session.commit()
            record_activity(
                self.session,
                event_type=f"agent.{action}.direct",
                message=f"{action.capitalize()}d directly for {agent.name}.",
                agent_id=agent.id,
            )
            record_activity(
                self.session,
                event_type="agent.wakeup.sent",
                message=f"Wakeup message sent to {agent.name}.",
                agent_id=agent.id,
            )
            await self.session.commit()
            self.logger.info(
                "agent.provision.success action=%s agent_id=%s",
                action,
                agent.id,
            )
        except OpenClawGatewayError as exc:
            self.record_instruction_failure(
                self.session,
                agent,
                str(exc),
                action,
            )
            await self.session.commit()
            self.logger.error(
                "agent.provision.gateway_error action=%s agent_id=%s error=%s",
                action,
                agent.id,
                str(exc),
            )
            if raise_gateway_errors:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Gateway {action} failed: {exc}",
                ) from exc
        except (OSError, RuntimeError, ValueError) as exc:  # pragma: no cover
            self.record_instruction_failure(
                self.session,
                agent,
                str(exc),
                action,
            )
            await self.session.commit()
            self.logger.critical(
                "agent.provision.runtime_error action=%s agent_id=%s error=%s",
                action,
                agent.id,
                str(exc),
            )
            if raise_gateway_errors:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Unexpected error {action}ing agent provisioning.",
                ) from exc

    async def provision_new_agent(
        self,
        *,
        agent: Agent,
        board: Board,
        gateway: Gateway,
        auth_token: str,
        user: User | None,
        force_bootstrap: bool,
    ) -> None:
        await self._apply_gateway_provisioning(
            agent=agent,
            target=AgentUpdateProvisionTarget(is_main_agent=False, board=board, gateway=gateway),
            auth_token=auth_token,
            user=user,
            action="provision",
            wakeup_verb="provisioned",
            force_bootstrap=force_bootstrap,
            raise_gateway_errors=True,
        )

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
            )

        if make_main is None and agent.board_id is None and main_gateway is not None:
            return AgentUpdateProvisionTarget(
                is_main_agent=True,
                board=None,
                gateway=main_gateway,
            )

        if agent.board_id is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="board_id is required for non-main agents",
            )
        board = await self.require_board(agent.board_id)
        gateway, _client_config = await self.require_gateway(board)
        return AgentUpdateProvisionTarget(
            is_main_agent=False,
            board=board,
            gateway=gateway,
        )

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
        await self._apply_gateway_provisioning(
            agent=agent,
            target=request.target,
            auth_token=request.raw_token,
            user=request.user,
            action="update",
            wakeup_verb="updated",
            force_bootstrap=request.force_bootstrap,
            raise_gateway_errors=True,
        )

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
        gateway, _client_config = await self.require_gateway(board)
        data: dict[str, Any] = {
            "name": payload.name,
            "board_id": board.id,
            "gateway_id": gateway.id,
            "heartbeat_config": DEFAULT_HEARTBEAT_CONFIG.copy(),
        }
        agent, raw_token = await self.persist_new_agent(data=data)
        await self.provision_new_agent(
            agent=agent,
            board=board,
            gateway=gateway,
            auth_token=raw_token,
            user=actor.user,
            force_bootstrap=False,
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
        gateway, _client_config = await self.require_gateway(board)
        await self.provision_new_agent(
            agent=agent,
            board=board,
            gateway=gateway,
            auth_token=raw_token,
            user=user,
            force_bootstrap=False,
        )

    async def ensure_heartbeat_session_key(
        self,
        *,
        agent: Agent,
        actor: ActorContextLike,
    ) -> None:
        _ = actor
        if agent.board_id is None:
            return
        desired = self.resolve_session_key(agent)
        existing = (agent.openclaw_session_id or "").strip()
        if existing == desired:
            return
        agent.openclaw_session_id = desired
        self.session.add(agent)
        await self.session.commit()
        await self.session.refresh(agent)

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
        gateway, _client_config = await self.require_gateway(board)
        data = payload.model_dump()
        data["gateway_id"] = gateway.id
        requested_name = (data.get("name") or "").strip()
        await self.ensure_unique_agent_name(
            board=board,
            gateway=gateway,
            requested_name=requested_name,
        )
        agent, raw_token = await self.persist_new_agent(data=data)
        await self.provision_new_agent(
            agent=agent,
            board=board,
            gateway=gateway,
            auth_token=raw_token,
            user=actor.user if actor.actor_type == "user" else None,
            force_bootstrap=False,
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

        gateway: Gateway | None = None
        client_config: GatewayClientConfig | None = None
        workspace_path: str | None = None

        if agent.board_id is None:
            # Gateway-main agents are not tied to a board; resolve via agent.gateway_id.
            gateway = await Gateway.objects.by_id(agent.gateway_id).first(self.session)
            if gateway and gateway.url:
                client_config = GatewayClientConfig(url=gateway.url, token=gateway.token)
                try:
                    workspace_path = await OpenClawGatewayProvisioner().delete_agent_lifecycle(
                        agent=agent,
                        gateway=gateway,
                    )
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
        else:
            board = await self.require_board(str(agent.board_id))
            gateway, client_config = await self.require_gateway(board)
            try:
                workspace_path = await OpenClawGatewayProvisioner().delete_agent_lifecycle(
                    agent=agent,
                    gateway=gateway,
                )
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
            # Notify the gateway-main agent about cleanup for board-scoped deletes.
            # Skip when deleting the gateway-main agent itself.
            if gateway is None or client_config is None or agent.board_id is None:
                raise ValueError("skip main agent cleanup notification")
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
