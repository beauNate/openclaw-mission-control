from __future__ import annotations

import asyncio
import random
import re
from collections.abc import Awaitable, Callable
from typing import TypeVar
from uuid import UUID, uuid4

from sqlalchemy import func
from sqlmodel import col, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.agent_tokens import generate_agent_token, hash_agent_token, verify_agent_token
from app.core.time import utcnow
from app.integrations.openclaw_gateway import GatewayConfig as GatewayClientConfig
from app.integrations.openclaw_gateway import OpenClawGatewayError, openclaw_call
from app.models.agents import Agent
from app.models.board_memory import BoardMemory
from app.models.boards import Board
from app.models.gateways import Gateway
from app.models.users import User
from app.schemas.gateways import GatewayTemplatesSyncError, GatewayTemplatesSyncResult
from app.services.agent_provisioning import provision_agent, provision_main_agent

_TOOLS_KV_RE = re.compile(r"^(?P<key>[A-Z0-9_]+)=(?P<value>.*)$")

T = TypeVar("T")


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or uuid4().hex


def _is_transient_gateway_error(exc: Exception) -> bool:
    if not isinstance(exc, OpenClawGatewayError):
        return False
    message = str(exc).lower()
    if not message:
        return False
    if "unsupported file" in message:
        return False
    if "connect call failed" in message or "connection refused" in message:
        return True
    if "errno 111" in message or "econnrefused" in message:
        return True
    if "did not receive a valid http response" in message:
        return True
    if "no route to host" in message or "network is unreachable" in message:
        return True
    if "host is down" in message or "name or service not known" in message:
        return True
    if "received 1012" in message or "service restart" in message:
        return True
    if "http 503" in message or ("503" in message and "websocket" in message):
        return True
    if "http 502" in message or "http 504" in message:
        return True
    if "temporar" in message:
        return True
    if "timeout" in message or "timed out" in message:
        return True
    if "connection closed" in message or "connection reset" in message:
        return True
    return False


class _GatewayBackoff:
    def __init__(
        self,
        *,
        timeout_s: float = 10 * 60,
        base_delay_s: float = 0.75,
        max_delay_s: float = 30.0,
        jitter: float = 0.2,
    ) -> None:
        self._timeout_s = timeout_s
        self._base_delay_s = base_delay_s
        self._max_delay_s = max_delay_s
        self._jitter = jitter
        self._delay_s = base_delay_s

    def reset(self) -> None:
        self._delay_s = self._base_delay_s

    async def run(self, fn: Callable[[], Awaitable[T]]) -> T:
        # Use per-call deadlines so long-running syncs can still tolerate a later
        # gateway restart without having an already-expired retry window.
        deadline_s = asyncio.get_running_loop().time() + self._timeout_s
        while True:
            try:
                value = await fn()
                self.reset()
                return value
            except Exception as exc:
                if not _is_transient_gateway_error(exc):
                    raise
                now = asyncio.get_running_loop().time()
                remaining = deadline_s - now
                if remaining <= 0:
                    raise TimeoutError(
                        "Gateway unreachable after 10 minutes (template sync timeout). "
                        f"Last error: {exc}"
                    ) from exc

                sleep_s = min(self._delay_s, remaining)
                if self._jitter:
                    sleep_s *= 1.0 + random.uniform(-self._jitter, self._jitter)
                sleep_s = max(0.0, min(sleep_s, remaining))
                await asyncio.sleep(sleep_s)
                self._delay_s = min(self._delay_s * 2.0, self._max_delay_s)


async def _with_gateway_retry(
    fn: Callable[[], Awaitable[T]],
    *,
    backoff: _GatewayBackoff,
) -> T:
    return await backoff.run(fn)


def _agent_id_from_session_key(session_key: str | None) -> str | None:
    value = (session_key or "").strip()
    if not value:
        return None
    if not value.startswith("agent:"):
        return None
    parts = value.split(":")
    if len(parts) < 2:
        return None
    agent_id = parts[1].strip()
    return agent_id or None


def _extract_agent_id(payload: object) -> str | None:
    def _from_list(items: object) -> str | None:
        if not isinstance(items, list):
            return None
        for item in items:
            if isinstance(item, str) and item.strip():
                return item.strip()
            if not isinstance(item, dict):
                continue
            for key in ("id", "agentId", "agent_id"):
                raw = item.get(key)
                if isinstance(raw, str) and raw.strip():
                    return raw.strip()
        return None

    if isinstance(payload, list):
        return _from_list(payload)
    if not isinstance(payload, dict):
        return None
    for key in ("defaultId", "default_id", "defaultAgentId", "default_agent_id"):
        raw = payload.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    for key in ("agents", "items", "list", "data"):
        agent_id = _from_list(payload.get(key))
        if agent_id:
            return agent_id
    return None


def _gateway_agent_id(agent: Agent) -> str:
    session_key = agent.openclaw_session_id or ""
    if session_key.startswith("agent:"):
        parts = session_key.split(":")
        if len(parts) >= 2 and parts[1]:
            return parts[1]
    return _slugify(agent.name)


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
    config: GatewayClientConfig,
    backoff: _GatewayBackoff | None = None,
) -> str | None:
    try:

        async def _do_get() -> object:
            return await openclaw_call(
                "agents.files.get",
                {"agentId": agent_gateway_id, "name": name},
                config=config,
            )

        payload = await (backoff.run(_do_get) if backoff else _do_get())
    except OpenClawGatewayError:
        return None
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        # Common shapes:
        # - {"name": "...", "content": "..."}
        # - {"file": {"name": "...", "content": "..." }}
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
    config: GatewayClientConfig,
    backoff: _GatewayBackoff | None = None,
) -> str | None:
    tools = await _get_agent_file(
        agent_gateway_id=agent_gateway_id,
        name="TOOLS.md",
        config=config,
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


async def _gateway_default_agent_id(
    config: GatewayClientConfig,
    *,
    fallback_session_key: str | None = None,
    backoff: _GatewayBackoff | None = None,
) -> str | None:
    try:

        async def _do_list() -> object:
            return await openclaw_call("agents.list", config=config)

        payload = await (backoff.run(_do_list) if backoff else _do_list())
        agent_id = _extract_agent_id(payload)
        if agent_id:
            return agent_id
    except OpenClawGatewayError:
        pass
    return _agent_id_from_session_key(fallback_session_key)


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


async def sync_gateway_templates(
    session: AsyncSession,
    gateway: Gateway,
    *,
    user: User | None,
    include_main: bool = True,
    reset_sessions: bool = False,
    rotate_tokens: bool = False,
    force_bootstrap: bool = False,
    board_id: UUID | None = None,
) -> GatewayTemplatesSyncResult:
    result = GatewayTemplatesSyncResult(
        gateway_id=gateway.id,
        include_main=include_main,
        reset_sessions=reset_sessions,
        agents_updated=0,
        agents_skipped=0,
        main_updated=False,
    )
    if not gateway.url:
        result.errors.append(
            GatewayTemplatesSyncError(message="Gateway URL is not configured for this gateway.")
        )
        return result

    client_config = GatewayClientConfig(url=gateway.url, token=gateway.token)
    backoff = _GatewayBackoff(timeout_s=10 * 60)

    # First, wait for the gateway to be reachable (e.g. while it is restarting).
    try:

        async def _do_ping() -> object:
            return await openclaw_call("agents.list", config=client_config)

        await backoff.run(_do_ping)
    except TimeoutError as exc:
        result.errors.append(GatewayTemplatesSyncError(message=str(exc)))
        return result
    except OpenClawGatewayError as exc:
        result.errors.append(GatewayTemplatesSyncError(message=str(exc)))
        return result

    boards = list(await session.exec(select(Board).where(col(Board.gateway_id) == gateway.id)))
    boards_by_id = {board.id: board for board in boards}
    if board_id is not None:
        board = boards_by_id.get(board_id)
        if board is None:
            result.errors.append(
                GatewayTemplatesSyncError(
                    board_id=board_id,
                    message="Board does not belong to this gateway.",
                )
            )
            return result
        boards_by_id = {board_id: board}

    paused_board_ids = await _paused_board_ids(session, list(boards_by_id.keys()))

    if boards_by_id:
        agents = list(
            await session.exec(
                select(Agent)
                .where(col(Agent.board_id).in_(list(boards_by_id.keys())))
                .order_by(col(Agent.created_at).asc())
            )
        )
    else:
        agents = []

    for agent in agents:
        board = boards_by_id.get(agent.board_id) if agent.board_id is not None else None
        if board is None:
            result.agents_skipped += 1
            result.errors.append(
                GatewayTemplatesSyncError(
                    agent_id=agent.id,
                    agent_name=agent.name,
                    board_id=agent.board_id,
                    message="Skipping agent: board not found for agent.",
                )
            )
            continue

        if board.id in paused_board_ids:
            result.agents_skipped += 1
            continue

        agent_gateway_id = _gateway_agent_id(agent)
        try:
            auth_token = await _get_existing_auth_token(
                agent_gateway_id=agent_gateway_id,
                config=client_config,
                backoff=backoff,
            )
        except TimeoutError as exc:
            result.errors.append(
                GatewayTemplatesSyncError(
                    agent_id=agent.id,
                    agent_name=agent.name,
                    board_id=board.id,
                    message=str(exc),
                )
            )
            return result

        if not auth_token:
            if not rotate_tokens:
                result.agents_skipped += 1
                result.errors.append(
                    GatewayTemplatesSyncError(
                        agent_id=agent.id,
                        agent_name=agent.name,
                        board_id=board.id,
                        message="Skipping agent: unable to read AUTH_TOKEN from TOOLS.md (run with rotate_tokens=true to re-key).",
                    )
                )
                continue
            raw_token = generate_agent_token()
            agent.agent_token_hash = hash_agent_token(raw_token)
            agent.updated_at = utcnow()
            session.add(agent)
            await session.commit()
            await session.refresh(agent)
            auth_token = raw_token

        if agent.agent_token_hash and not verify_agent_token(auth_token, agent.agent_token_hash):
            # Do not block template sync on token drift; optionally re-key.
            if rotate_tokens:
                raw_token = generate_agent_token()
                agent.agent_token_hash = hash_agent_token(raw_token)
                agent.updated_at = utcnow()
                session.add(agent)
                await session.commit()
                await session.refresh(agent)
                auth_token = raw_token
            else:
                result.errors.append(
                    GatewayTemplatesSyncError(
                        agent_id=agent.id,
                        agent_name=agent.name,
                        board_id=board.id,
                        message="Warning: AUTH_TOKEN in TOOLS.md does not match backend token hash (agent auth may be broken).",
                    )
                )

        try:

            async def _do_provision() -> None:
                await provision_agent(
                    agent,
                    board,
                    gateway,
                    auth_token,
                    user,
                    action="update",
                    force_bootstrap=force_bootstrap,
                    reset_session=reset_sessions,
                )

            await _with_gateway_retry(_do_provision, backoff=backoff)
            result.agents_updated += 1
        except TimeoutError as exc:  # pragma: no cover - gateway/network dependent
            result.agents_skipped += 1
            result.errors.append(
                GatewayTemplatesSyncError(
                    agent_id=agent.id,
                    agent_name=agent.name,
                    board_id=board.id,
                    message=str(exc),
                )
            )
            return result
        except Exception as exc:  # pragma: no cover - gateway/network dependent
            result.agents_skipped += 1
            result.errors.append(
                GatewayTemplatesSyncError(
                    agent_id=agent.id,
                    agent_name=agent.name,
                    board_id=board.id,
                    message=f"Failed to sync templates: {exc}",
                )
            )

    if include_main:
        main_agent = (
            await session.exec(
                select(Agent).where(col(Agent.openclaw_session_id) == gateway.main_session_key)
            )
        ).first()
        if main_agent is None:
            result.errors.append(
                GatewayTemplatesSyncError(
                    message="Gateway main agent record not found; skipping main agent template sync.",
                )
            )
            return result

        try:
            main_gateway_agent_id = await _gateway_default_agent_id(
                client_config,
                fallback_session_key=gateway.main_session_key,
                backoff=backoff,
            )
        except TimeoutError as exc:
            result.errors.append(
                GatewayTemplatesSyncError(
                    agent_id=main_agent.id,
                    agent_name=main_agent.name,
                    message=str(exc),
                )
            )
            return result
        if not main_gateway_agent_id:
            result.errors.append(
                GatewayTemplatesSyncError(
                    agent_id=main_agent.id,
                    agent_name=main_agent.name,
                    message="Unable to resolve gateway default agent id for main agent.",
                )
            )
            return result

        try:
            main_token = await _get_existing_auth_token(
                agent_gateway_id=main_gateway_agent_id,
                config=client_config,
                backoff=backoff,
            )
        except TimeoutError as exc:
            result.errors.append(
                GatewayTemplatesSyncError(
                    agent_id=main_agent.id,
                    agent_name=main_agent.name,
                    message=str(exc),
                )
            )
            return result
        if not main_token:
            if rotate_tokens:
                raw_token = generate_agent_token()
                main_agent.agent_token_hash = hash_agent_token(raw_token)
                main_agent.updated_at = utcnow()
                session.add(main_agent)
                await session.commit()
                await session.refresh(main_agent)
                main_token = raw_token
            else:
                result.errors.append(
                    GatewayTemplatesSyncError(
                        agent_id=main_agent.id,
                        agent_name=main_agent.name,
                        message="Skipping main agent: unable to read AUTH_TOKEN from TOOLS.md.",
                    )
                )
                return result

        if main_agent.agent_token_hash and not verify_agent_token(
            main_token, main_agent.agent_token_hash
        ):
            if rotate_tokens:
                raw_token = generate_agent_token()
                main_agent.agent_token_hash = hash_agent_token(raw_token)
                main_agent.updated_at = utcnow()
                session.add(main_agent)
                await session.commit()
                await session.refresh(main_agent)
                main_token = raw_token
            else:
                result.errors.append(
                    GatewayTemplatesSyncError(
                        agent_id=main_agent.id,
                        agent_name=main_agent.name,
                        message="Warning: AUTH_TOKEN in TOOLS.md does not match backend token hash (main agent auth may be broken).",
                    )
                )

        try:

            async def _do_provision_main() -> None:
                await provision_main_agent(
                    main_agent,
                    gateway,
                    main_token,
                    user,
                    action="update",
                    force_bootstrap=force_bootstrap,
                    reset_session=reset_sessions,
                )

            await _with_gateway_retry(_do_provision_main, backoff=backoff)
            result.main_updated = True
        except TimeoutError as exc:  # pragma: no cover - gateway/network dependent
            result.errors.append(
                GatewayTemplatesSyncError(
                    agent_id=main_agent.id,
                    agent_name=main_agent.name,
                    message=str(exc),
                )
            )
            return result
        except Exception as exc:  # pragma: no cover - gateway/network dependent
            result.errors.append(
                GatewayTemplatesSyncError(
                    agent_id=main_agent.id,
                    agent_name=main_agent.name,
                    message=f"Failed to sync main agent templates: {exc}",
                )
            )

    return result
