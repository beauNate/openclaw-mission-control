"""DB-backed OpenClaw provisioning orchestration.

Layering:
- `app.services.openclaw.provisioning` contains gateway-only lifecycle operations (no DB calls).
- This module builds on top of that layer using AsyncSession for token rotation, lead-agent records,
  and bulk template synchronization.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar
from uuid import UUID

from sqlalchemy import func
from sqlmodel import col, select

from app.core.agent_tokens import generate_agent_token, hash_agent_token, verify_agent_token
from app.core.time import utcnow
from app.integrations.openclaw_gateway import GatewayConfig as GatewayClientConfig
from app.integrations.openclaw_gateway import OpenClawGatewayError
from app.models.agents import Agent
from app.models.board_memory import BoardMemory
from app.models.boards import Board
from app.models.gateways import Gateway
from app.schemas.gateways import GatewayTemplatesSyncError, GatewayTemplatesSyncResult
from app.services.openclaw.constants import (
    _NON_TRANSIENT_GATEWAY_ERROR_MARKERS,
    _SECURE_RANDOM,
    _TOOLS_KV_RE,
    _TRANSIENT_GATEWAY_ERROR_MARKERS,
    DEFAULT_HEARTBEAT_CONFIG,
)
from app.services.openclaw.internal import agent_key as _agent_key
from app.services.openclaw.provisioning import (
    OpenClawGatewayControlPlane,
    OpenClawGatewayProvisioner,
)
from app.services.openclaw.shared import GatewayAgentIdentity
from app.services.organizations import get_org_owner_user

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from sqlmodel.ext.asyncio.session import AsyncSession

    from app.models.users import User


_T = TypeVar("_T")


@dataclass(frozen=True)
class GatewayTemplateSyncOptions:
    """Runtime options controlling gateway template synchronization."""

    user: User | None
    include_main: bool = True
    reset_sessions: bool = False
    rotate_tokens: bool = False
    force_bootstrap: bool = False
    board_id: UUID | None = None


@dataclass(frozen=True, slots=True)
class LeadAgentOptions:
    """Optional overrides for board-lead provisioning behavior."""

    agent_name: str | None = None
    identity_profile: dict[str, str] | None = None
    action: str = "provision"


@dataclass(frozen=True, slots=True)
class LeadAgentRequest:
    """Inputs required to ensure or provision a board lead agent."""

    board: Board
    gateway: Gateway
    config: GatewayClientConfig
    user: User | None
    options: LeadAgentOptions = field(default_factory=LeadAgentOptions)


class OpenClawProvisioningService:
    """DB-backed provisioning workflows (bulk template sync, lead-agent record)."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._gateway = OpenClawGatewayProvisioner()

    @property
    def session(self) -> AsyncSession:
        return self._session

    @staticmethod
    def lead_session_key(board: Board) -> str:
        return f"agent:lead-{board.id}:main"

    @staticmethod
    def lead_agent_name(_: Board) -> str:
        return "Lead Agent"

    async def ensure_board_lead_agent(
        self,
        *,
        request: LeadAgentRequest,
    ) -> tuple[Agent, bool]:
        """Ensure a board has a lead agent; return `(agent, created)`."""
        board = request.board
        config_options = request.options

        existing = (
            await self.session.exec(
                select(Agent)
                .where(Agent.board_id == board.id)
                .where(col(Agent.is_board_lead).is_(True)),
            )
        ).first()
        if existing:
            desired_name = config_options.agent_name or self.lead_agent_name(board)
            changed = False
            if existing.name != desired_name:
                existing.name = desired_name
                changed = True
            if existing.gateway_id != request.gateway.id:
                existing.gateway_id = request.gateway.id
                changed = True
            desired_session_key = self.lead_session_key(board)
            if existing.openclaw_session_id != desired_session_key:
                existing.openclaw_session_id = desired_session_key
                changed = True
            if changed:
                existing.updated_at = utcnow()
                self.session.add(existing)
                await self.session.commit()
                await self.session.refresh(existing)
            return existing, False

        merged_identity_profile: dict[str, Any] = {
            "role": "Board Lead",
            "communication_style": "direct, concise, practical",
            "emoji": ":gear:",
        }
        if config_options.identity_profile:
            merged_identity_profile.update(
                {
                    key: value.strip()
                    for key, value in config_options.identity_profile.items()
                    if value.strip()
                },
            )

        agent = Agent(
            name=config_options.agent_name or self.lead_agent_name(board),
            status="provisioning",
            board_id=board.id,
            gateway_id=request.gateway.id,
            is_board_lead=True,
            heartbeat_config=DEFAULT_HEARTBEAT_CONFIG.copy(),
            identity_profile=merged_identity_profile,
            openclaw_session_id=self.lead_session_key(board),
            provision_requested_at=utcnow(),
            provision_action=config_options.action,
        )
        raw_token = generate_agent_token()
        agent.agent_token_hash = hash_agent_token(raw_token)
        self.session.add(agent)
        await self.session.commit()
        await self.session.refresh(agent)

        # Strict behavior: provisioning errors surface to the caller. The DB row exists
        # so a later retry can succeed with the same deterministic identity/session key.
        await self._gateway.apply_agent_lifecycle(
            agent=agent,
            gateway=request.gateway,
            board=board,
            auth_token=raw_token,
            user=request.user,
            action=config_options.action,
            wake=True,
            deliver_wakeup=True,
        )

        agent.status = "online"
        agent.provision_requested_at = None
        agent.provision_action = None
        agent.updated_at = utcnow()
        self.session.add(agent)
        await self.session.commit()
        await self.session.refresh(agent)

        return agent, True

    async def sync_gateway_templates(
        self,
        gateway: Gateway,
        options: GatewayTemplateSyncOptions,
    ) -> GatewayTemplatesSyncResult:
        """Synchronize AGENTS/TOOLS/etc templates to gateway-connected agents."""
        template_user = options.user
        if template_user is None:
            template_user = await get_org_owner_user(
                self.session,
                organization_id=gateway.organization_id,
            )
            options = GatewayTemplateSyncOptions(
                user=template_user,
                include_main=options.include_main,
                reset_sessions=options.reset_sessions,
                rotate_tokens=options.rotate_tokens,
                force_bootstrap=options.force_bootstrap,
                board_id=options.board_id,
            )

        result = _base_result(
            gateway,
            include_main=options.include_main,
            reset_sessions=options.reset_sessions,
        )
        if not gateway.url:
            _append_sync_error(
                result,
                message="Gateway URL is not configured for this gateway.",
            )
            return result

        control_plane = OpenClawGatewayControlPlane(
            GatewayClientConfig(url=gateway.url, token=gateway.token),
        )
        ctx = _SyncContext(
            session=self.session,
            gateway=gateway,
            control_plane=control_plane,
            backoff=_GatewayBackoff(timeout_s=10 * 60, timeout_context="template sync"),
            options=options,
            provisioner=self._gateway,
        )
        if not await _ping_gateway(ctx, result):
            return result

        boards = await Board.objects.filter_by(gateway_id=gateway.id).all(self.session)
        boards_by_id = _boards_by_id(boards, board_id=options.board_id)
        if boards_by_id is None:
            _append_sync_error(
                result,
                message="Board does not belong to this gateway.",
            )
            return result
        paused_board_ids = await _paused_board_ids(self.session, list(boards_by_id.keys()))
        if boards_by_id:
            agents = await (
                Agent.objects.by_field_in("board_id", list(boards_by_id.keys()))
                .order_by(col(Agent.created_at).asc())
                .all(self.session)
            )
        else:
            agents = []

        stop_sync = False
        for agent in agents:
            board = boards_by_id.get(agent.board_id) if agent.board_id is not None else None
            if board is None:
                result.agents_skipped += 1
                _append_sync_error(
                    result,
                    agent=agent,
                    message="Skipping agent: board not found for agent.",
                )
                continue
            if board.id in paused_board_ids:
                result.agents_skipped += 1
                continue
            stop_sync = await _sync_one_agent(ctx, result, agent, board)
            if stop_sync:
                break

        if not stop_sync and options.include_main:
            await _sync_main_agent(ctx, result)
        return result


@dataclass(frozen=True)
class _SyncContext:
    session: AsyncSession
    gateway: Gateway
    control_plane: OpenClawGatewayControlPlane
    backoff: _GatewayBackoff
    options: GatewayTemplateSyncOptions
    provisioner: OpenClawGatewayProvisioner


def _is_transient_gateway_error(exc: Exception) -> bool:
    if not isinstance(exc, OpenClawGatewayError):
        return False
    message = str(exc).lower()
    if not message:
        return False
    if any(marker in message for marker in _NON_TRANSIENT_GATEWAY_ERROR_MARKERS):
        return False
    return ("503" in message and "websocket" in message) or any(
        marker in message for marker in _TRANSIENT_GATEWAY_ERROR_MARKERS
    )


def _gateway_timeout_message(
    exc: OpenClawGatewayError,
    *,
    timeout_s: float,
    context: str,
) -> str:
    rounded_timeout = int(timeout_s)
    timeout_text = f"{rounded_timeout} seconds"
    if rounded_timeout >= 120:
        timeout_text = f"{rounded_timeout // 60} minutes"
    return f"Gateway unreachable after {timeout_text} ({context} timeout). Last error: {exc}"


class _GatewayBackoff:
    def __init__(
        self,
        *,
        timeout_s: float = 10 * 60,
        base_delay_s: float = 0.75,
        max_delay_s: float = 30.0,
        jitter: float = 0.2,
        timeout_context: str = "gateway operation",
    ) -> None:
        self._timeout_s = timeout_s
        self._base_delay_s = base_delay_s
        self._max_delay_s = max_delay_s
        self._jitter = jitter
        self._timeout_context = timeout_context
        self._delay_s = base_delay_s

    def reset(self) -> None:
        self._delay_s = self._base_delay_s

    @staticmethod
    async def _attempt(
        fn: Callable[[], Awaitable[_T]],
    ) -> tuple[_T | None, OpenClawGatewayError | None]:
        try:
            return await fn(), None
        except OpenClawGatewayError as exc:
            return None, exc

    async def run(self, fn: Callable[[], Awaitable[_T]]) -> _T:
        deadline_s = asyncio.get_running_loop().time() + self._timeout_s
        while True:
            value, error = await self._attempt(fn)
            if error is not None:
                exc = error
                if not _is_transient_gateway_error(exc):
                    raise exc
                now = asyncio.get_running_loop().time()
                remaining = deadline_s - now
                if remaining <= 0:
                    raise TimeoutError(
                        _gateway_timeout_message(
                            exc,
                            timeout_s=self._timeout_s,
                            context=self._timeout_context,
                        ),
                    ) from exc

                sleep_s = min(self._delay_s, remaining)
                if self._jitter:
                    sleep_s *= 1.0 + _SECURE_RANDOM.uniform(
                        -self._jitter,
                        self._jitter,
                    )
                sleep_s = max(0.0, min(sleep_s, remaining))
                await asyncio.sleep(sleep_s)
                self._delay_s = min(self._delay_s * 2.0, self._max_delay_s)
                continue
            self.reset()
            if value is None:
                msg = "Gateway retry produced no value without an error"
                raise RuntimeError(msg)
            return value


async def _with_gateway_retry(
    fn: Callable[[], Awaitable[_T]],
    *,
    backoff: _GatewayBackoff,
) -> _T:
    return await backoff.run(fn)


def _parse_tools_md(content: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in content.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _TOOLS_KV_RE.match(line)
        if not match:
            continue
        values[match.group("key")] = match.group("value").strip()
    return values


async def _get_agent_file(
    *,
    agent_gateway_id: str,
    name: str,
    control_plane: OpenClawGatewayControlPlane,
    backoff: _GatewayBackoff | None = None,
) -> str | None:
    try:

        async def _do_get() -> object:
            return await control_plane.get_agent_file_payload(agent_id=agent_gateway_id, name=name)

        payload = await (backoff.run(_do_get) if backoff else _do_get())
    except OpenClawGatewayError:
        return None
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        content = payload.get("content")
        if isinstance(content, str):
            return content
        file_obj = payload.get("file")
        if isinstance(file_obj, dict):
            nested = file_obj.get("content")
            if isinstance(nested, str):
                return nested
    return None


async def _get_existing_auth_token(
    *,
    agent_gateway_id: str,
    control_plane: OpenClawGatewayControlPlane,
    backoff: _GatewayBackoff | None = None,
) -> str | None:
    tools = await _get_agent_file(
        agent_gateway_id=agent_gateway_id,
        name="TOOLS.md",
        control_plane=control_plane,
        backoff=backoff,
    )
    if not tools:
        return None
    values = _parse_tools_md(tools)
    token = values.get("AUTH_TOKEN")
    if not token:
        return None
    token = token.strip()
    return token or None


async def _paused_board_ids(session: AsyncSession, board_ids: list[UUID]) -> set[UUID]:
    if not board_ids:
        return set()

    commands = {"/pause", "/resume"}
    statement = (
        select(BoardMemory.board_id, BoardMemory.content)
        .where(col(BoardMemory.board_id).in_(board_ids))
        .where(col(BoardMemory.is_chat).is_(True))
        .where(func.lower(func.trim(col(BoardMemory.content))).in_(commands))
        .order_by(col(BoardMemory.board_id), col(BoardMemory.created_at).desc())
        # Postgres: DISTINCT ON (board_id) to get latest command per board.
        .distinct(col(BoardMemory.board_id))
    )

    paused: set[UUID] = set()
    for board_id, content in await session.exec(statement):
        cmd = (content or "").strip().lower()
        if cmd == "/pause":
            paused.add(board_id)
    return paused


def _append_sync_error(
    result: GatewayTemplatesSyncResult,
    *,
    message: str,
    agent: Agent | None = None,
    board: Board | None = None,
) -> None:
    result.errors.append(
        GatewayTemplatesSyncError(
            agent_id=agent.id if agent else None,
            agent_name=agent.name if agent else None,
            board_id=board.id if board else None,
            message=message,
        ),
    )


async def _rotate_agent_token(session: AsyncSession, agent: Agent) -> str:
    token = generate_agent_token()
    agent.agent_token_hash = hash_agent_token(token)
    agent.updated_at = utcnow()
    session.add(agent)
    await session.commit()
    await session.refresh(agent)
    return token


async def _ping_gateway(ctx: _SyncContext, result: GatewayTemplatesSyncResult) -> bool:
    try:

        async def _do_ping() -> object:
            return await ctx.control_plane.health()

        await ctx.backoff.run(_do_ping)
    except (TimeoutError, OpenClawGatewayError) as exc:
        _append_sync_error(result, message=str(exc))
        return False
    else:
        return True


def _base_result(
    gateway: Gateway,
    *,
    include_main: bool,
    reset_sessions: bool,
) -> GatewayTemplatesSyncResult:
    return GatewayTemplatesSyncResult(
        gateway_id=gateway.id,
        include_main=include_main,
        reset_sessions=reset_sessions,
        agents_updated=0,
        agents_skipped=0,
        main_updated=False,
    )


def _boards_by_id(
    boards: list[Board],
    *,
    board_id: UUID | None,
) -> dict[UUID, Board] | None:
    boards_by_id = {board.id: board for board in boards}
    if board_id is None:
        return boards_by_id
    board = boards_by_id.get(board_id)
    if board is None:
        return None
    return {board_id: board}


async def _resolve_agent_auth_token(
    ctx: _SyncContext,
    result: GatewayTemplatesSyncResult,
    agent: Agent,
    board: Board | None,
    *,
    agent_gateway_id: str,
) -> tuple[str | None, bool]:
    try:
        auth_token = await _get_existing_auth_token(
            agent_gateway_id=agent_gateway_id,
            control_plane=ctx.control_plane,
            backoff=ctx.backoff,
        )
    except TimeoutError as exc:
        _append_sync_error(result, agent=agent, board=board, message=str(exc))
        return None, True

    if not auth_token:
        if not ctx.options.rotate_tokens:
            result.agents_skipped += 1
            _append_sync_error(
                result,
                agent=agent,
                board=board,
                message=(
                    "Skipping agent: unable to read AUTH_TOKEN from TOOLS.md "
                    "(run with rotate_tokens=true to re-key)."
                ),
            )
            return None, False
        auth_token = await _rotate_agent_token(ctx.session, agent)

    if agent.agent_token_hash and not verify_agent_token(
        auth_token,
        agent.agent_token_hash,
    ):
        if ctx.options.rotate_tokens:
            auth_token = await _rotate_agent_token(ctx.session, agent)
        else:
            _append_sync_error(
                result,
                agent=agent,
                board=board,
                message=(
                    "Warning: AUTH_TOKEN in TOOLS.md does not match backend "
                    "token hash (agent auth may be broken)."
                ),
            )
    return auth_token, False


async def _sync_one_agent(
    ctx: _SyncContext,
    result: GatewayTemplatesSyncResult,
    agent: Agent,
    board: Board,
) -> bool:
    auth_token, fatal = await _resolve_agent_auth_token(
        ctx,
        result,
        agent,
        board,
        agent_gateway_id=_agent_key(agent),
    )
    if fatal:
        return True
    if not auth_token:
        return False
    try:

        async def _do_provision() -> bool:
            await ctx.provisioner.apply_agent_lifecycle(
                agent=agent,
                gateway=ctx.gateway,
                board=board,
                auth_token=auth_token,
                user=ctx.options.user,
                action="update",
                force_bootstrap=ctx.options.force_bootstrap,
                reset_session=ctx.options.reset_sessions,
                wake=False,
            )
            return True

        await _with_gateway_retry(_do_provision, backoff=ctx.backoff)
        result.agents_updated += 1
    except TimeoutError as exc:  # pragma: no cover - gateway/network dependent
        result.agents_skipped += 1
        _append_sync_error(result, agent=agent, board=board, message=str(exc))
        return True
    except (OSError, RuntimeError, ValueError) as exc:  # pragma: no cover
        result.agents_skipped += 1
        _append_sync_error(
            result,
            agent=agent,
            board=board,
            message=f"Failed to sync templates: {exc}",
        )
        return False
    else:
        return False


async def _sync_main_agent(
    ctx: _SyncContext,
    result: GatewayTemplatesSyncResult,
) -> bool:
    main_agent = (
        await Agent.objects.all()
        .filter(col(Agent.gateway_id) == ctx.gateway.id)
        .filter(col(Agent.board_id).is_(None))
        .first(ctx.session)
    )
    if main_agent is None:
        _append_sync_error(
            result,
            message="Gateway agent record not found; skipping gateway agent template sync.",
        )
        return True

    main_gateway_agent_id = GatewayAgentIdentity.openclaw_agent_id(ctx.gateway)
    token, fatal = await _resolve_agent_auth_token(
        ctx,
        result,
        main_agent,
        board=None,
        agent_gateway_id=main_gateway_agent_id,
    )
    if fatal:
        return True
    if not token:
        _append_sync_error(
            result,
            agent=main_agent,
            message="Skipping gateway agent: unable to read AUTH_TOKEN from TOOLS.md.",
        )
        return True
    stop_sync = False
    try:

        async def _do_provision_main() -> bool:
            await ctx.provisioner.apply_agent_lifecycle(
                agent=main_agent,
                gateway=ctx.gateway,
                board=None,
                auth_token=token,
                user=ctx.options.user,
                action="update",
                force_bootstrap=ctx.options.force_bootstrap,
                reset_session=ctx.options.reset_sessions,
                wake=False,
            )
            return True

        await _with_gateway_retry(_do_provision_main, backoff=ctx.backoff)
    except TimeoutError as exc:  # pragma: no cover - gateway/network dependent
        _append_sync_error(result, agent=main_agent, message=str(exc))
        stop_sync = True
    except (OSError, RuntimeError, ValueError) as exc:  # pragma: no cover
        _append_sync_error(
            result,
            agent=main_agent,
            message=f"Failed to sync gateway agent templates: {exc}",
        )
    else:
        result.main_updated = True
    return stop_sync
