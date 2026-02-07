from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy import func
from sqlmodel import col, select
from sqlmodel.ext.asyncio.session import AsyncSession
from sse_starlette.sse import EventSourceResponse

from app.api.deps import ActorContext, get_board_or_404, require_admin_auth, require_admin_or_agent
from app.core.auth import AuthContext
from app.core.config import settings
from app.core.time import utcnow
from app.db.pagination import paginate
from app.db.session import async_session_maker, get_session
from app.integrations.openclaw_gateway import GatewayConfig as GatewayClientConfig
from app.integrations.openclaw_gateway import OpenClawGatewayError, ensure_session, send_message
from app.models.agents import Agent
from app.models.board_group_memory import BoardGroupMemory
from app.models.board_groups import BoardGroup
from app.models.boards import Board
from app.models.gateways import Gateway
from app.schemas.board_group_memory import BoardGroupMemoryCreate, BoardGroupMemoryRead
from app.schemas.pagination import DefaultLimitOffsetPage
from app.services.mentions import extract_mentions, matches_agent_mention

router = APIRouter(tags=["board-group-memory"])

group_router = APIRouter(prefix="/board-groups/{group_id}/memory", tags=["board-group-memory"])
board_router = APIRouter(prefix="/boards/{board_id}/group-memory", tags=["board-group-memory"])


def _parse_since(value: str | None) -> datetime | None:
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
        return parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _serialize_memory(memory: BoardGroupMemory) -> dict[str, object]:
    return BoardGroupMemoryRead.model_validate(memory, from_attributes=True).model_dump(mode="json")


async def _gateway_config(session: AsyncSession, board: Board) -> GatewayClientConfig | None:
    if board.gateway_id is None:
        return None
    gateway = await session.get(Gateway, board.gateway_id)
    if gateway is None or not gateway.url:
        return None
    return GatewayClientConfig(url=gateway.url, token=gateway.token)


async def _send_agent_message(
    *,
    session_key: str,
    config: GatewayClientConfig,
    agent_name: str,
    message: str,
    deliver: bool = False,
) -> None:
    await ensure_session(session_key, config=config, label=agent_name)
    await send_message(message, session_key=session_key, config=config, deliver=deliver)


async def _fetch_memory_events(
    session: AsyncSession,
    board_group_id: UUID,
    since: datetime,
    is_chat: bool | None = None,
) -> list[BoardGroupMemory]:
    statement = (
        select(BoardGroupMemory).where(col(BoardGroupMemory.board_group_id) == board_group_id)
        # Old/invalid rows (empty/whitespace-only content) can exist; exclude them to
        # satisfy the NonEmptyStr response schema.
        .where(func.length(func.trim(col(BoardGroupMemory.content))) > 0)
    )
    if is_chat is not None:
        statement = statement.where(col(BoardGroupMemory.is_chat) == is_chat)
    statement = statement.where(col(BoardGroupMemory.created_at) >= since).order_by(
        col(BoardGroupMemory.created_at)
    )
    return list(await session.exec(statement))


async def _notify_group_memory_targets(
    *,
    session: AsyncSession,
    group: BoardGroup,
    memory: BoardGroupMemory,
    actor: ActorContext,
) -> None:
    if not memory.content:
        return

    tags = set(memory.tags or [])
    mentions = extract_mentions(memory.content)
    is_broadcast = "broadcast" in tags or "all" in mentions

    # Fetch group boards + agents.
    boards = list(await session.exec(select(Board).where(col(Board.board_group_id) == group.id)))
    if not boards:
        return
    board_by_id = {board.id: board for board in boards}
    board_ids = list(board_by_id.keys())
    agents = list(await session.exec(select(Agent).where(col(Agent.board_id).in_(board_ids))))

    targets: dict[str, Agent] = {}
    for agent in agents:
        if not agent.openclaw_session_id:
            continue
        if actor.actor_type == "agent" and actor.agent and agent.id == actor.agent.id:
            continue
        if is_broadcast:
            targets[str(agent.id)] = agent
            continue
        if agent.is_board_lead:
            targets[str(agent.id)] = agent
            continue
        if mentions and matches_agent_mention(agent, mentions):
            targets[str(agent.id)] = agent

    if not targets:
        return

    actor_name = "User"
    if actor.actor_type == "agent" and actor.agent:
        actor_name = actor.agent.name
    elif actor.user:
        actor_name = actor.user.preferred_name or actor.user.name or actor_name

    snippet = memory.content.strip()
    if len(snippet) > 800:
        snippet = f"{snippet[:797]}..."

    base_url = settings.base_url or "http://localhost:8000"

    for agent in targets.values():
        session_key = agent.openclaw_session_id
        if not session_key:
            continue
        board_id = agent.board_id
        if board_id is None:
            continue
        board = board_by_id.get(board_id)
        if board is None:
            continue
        config = await _gateway_config(session, board)
        if config is None:
            continue
        mentioned = matches_agent_mention(agent, mentions)
        if is_broadcast:
            header = "GROUP BROADCAST"
        elif mentioned:
            header = "GROUP CHAT MENTION"
        else:
            header = "GROUP CHAT"
        message = (
            f"{header}\n"
            f"Group: {group.name}\n"
            f"From: {actor_name}\n\n"
            f"{snippet}\n\n"
            "Reply via group chat (shared across linked boards):\n"
            f"POST {base_url}/api/v1/boards/{board.id}/group-memory\n"
            'Body: {"content":"...","tags":["chat"]}'
        )
        try:
            await _send_agent_message(
                session_key=session_key,
                config=config,
                agent_name=agent.name,
                message=message,
            )
        except OpenClawGatewayError:
            continue


@group_router.get("", response_model=DefaultLimitOffsetPage[BoardGroupMemoryRead])
async def list_board_group_memory(
    group_id: UUID,
    is_chat: bool | None = Query(default=None),
    session: AsyncSession = Depends(get_session),
    auth: AuthContext = Depends(require_admin_auth),
) -> DefaultLimitOffsetPage[BoardGroupMemoryRead]:
    group = await session.get(BoardGroup, group_id)
    if group is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    statement = (
        select(BoardGroupMemory).where(col(BoardGroupMemory.board_group_id) == group_id)
        # Old/invalid rows (empty/whitespace-only content) can exist; exclude them to
        # satisfy the NonEmptyStr response schema.
        .where(func.length(func.trim(col(BoardGroupMemory.content))) > 0)
    )
    if is_chat is not None:
        statement = statement.where(col(BoardGroupMemory.is_chat) == is_chat)
    statement = statement.order_by(col(BoardGroupMemory.created_at).desc())
    return await paginate(session, statement)


@group_router.get("/stream")
async def stream_board_group_memory(
    group_id: UUID,
    request: Request,
    since: str | None = Query(default=None),
    is_chat: bool | None = Query(default=None),
    session: AsyncSession = Depends(get_session),
    auth: AuthContext = Depends(require_admin_auth),
) -> EventSourceResponse:
    group = await session.get(BoardGroup, group_id)
    if group is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    since_dt = _parse_since(since) or utcnow()
    last_seen = since_dt

    async def event_generator() -> AsyncIterator[dict[str, str]]:
        nonlocal last_seen
        while True:
            if await request.is_disconnected():
                break
            async with async_session_maker() as s:
                memories = await _fetch_memory_events(
                    s,
                    group_id,
                    last_seen,
                    is_chat=is_chat,
                )
            for memory in memories:
                if memory.created_at > last_seen:
                    last_seen = memory.created_at
                payload = {"memory": _serialize_memory(memory)}
                yield {"event": "memory", "data": json.dumps(payload)}
            await asyncio.sleep(2)

    return EventSourceResponse(event_generator(), ping=15)


@group_router.post("", response_model=BoardGroupMemoryRead)
async def create_board_group_memory(
    group_id: UUID,
    payload: BoardGroupMemoryCreate,
    session: AsyncSession = Depends(get_session),
    auth: AuthContext = Depends(require_admin_auth),
) -> BoardGroupMemory:
    group = await session.get(BoardGroup, group_id)
    if group is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)

    actor = ActorContext(actor_type="user", user=auth.user)
    tags = set(payload.tags or [])
    is_chat = "chat" in tags
    mentions = extract_mentions(payload.content)
    should_notify = is_chat or "broadcast" in tags or "all" in mentions
    source = payload.source
    if should_notify and not source:
        if actor.actor_type == "agent" and actor.agent:
            source = actor.agent.name
        elif actor.user:
            source = actor.user.preferred_name or actor.user.name or "User"
    memory = BoardGroupMemory(
        board_group_id=group_id,
        content=payload.content,
        tags=payload.tags,
        is_chat=is_chat,
        source=source,
    )
    session.add(memory)
    await session.commit()
    await session.refresh(memory)
    if should_notify:
        await _notify_group_memory_targets(session=session, group=group, memory=memory, actor=actor)
    return memory


@board_router.get("", response_model=DefaultLimitOffsetPage[BoardGroupMemoryRead])
async def list_board_group_memory_for_board(
    is_chat: bool | None = Query(default=None),
    board: Board = Depends(get_board_or_404),
    session: AsyncSession = Depends(get_session),
    actor: ActorContext = Depends(require_admin_or_agent),
) -> DefaultLimitOffsetPage[BoardGroupMemoryRead]:
    if actor.actor_type == "agent" and actor.agent:
        if actor.agent.board_id and actor.agent.board_id != board.id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
    group_id = board.board_group_id
    if group_id is None:
        statement = select(BoardGroupMemory).where(col(BoardGroupMemory.id).is_(None))
        return await paginate(session, statement)

    statement = (
        select(BoardGroupMemory).where(col(BoardGroupMemory.board_group_id) == group_id)
        # Old/invalid rows (empty/whitespace-only content) can exist; exclude them to
        # satisfy the NonEmptyStr response schema.
        .where(func.length(func.trim(col(BoardGroupMemory.content))) > 0)
    )
    if is_chat is not None:
        statement = statement.where(col(BoardGroupMemory.is_chat) == is_chat)
    statement = statement.order_by(col(BoardGroupMemory.created_at).desc())
    return await paginate(session, statement)


@board_router.get("/stream")
async def stream_board_group_memory_for_board(
    request: Request,
    board: Board = Depends(get_board_or_404),
    actor: ActorContext = Depends(require_admin_or_agent),
    since: str | None = Query(default=None),
    is_chat: bool | None = Query(default=None),
) -> EventSourceResponse:
    if actor.actor_type == "agent" and actor.agent:
        if actor.agent.board_id and actor.agent.board_id != board.id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
    group_id = board.board_group_id
    since_dt = _parse_since(since) or utcnow()
    last_seen = since_dt

    async def event_generator() -> AsyncIterator[dict[str, str]]:
        nonlocal last_seen
        while True:
            if await request.is_disconnected():
                break
            if group_id is None:
                await asyncio.sleep(2)
                continue
            async with async_session_maker() as session:
                memories = await _fetch_memory_events(
                    session,
                    group_id,
                    last_seen,
                    is_chat=is_chat,
                )
            for memory in memories:
                if memory.created_at > last_seen:
                    last_seen = memory.created_at
                payload = {"memory": _serialize_memory(memory)}
                yield {"event": "memory", "data": json.dumps(payload)}
            await asyncio.sleep(2)

    return EventSourceResponse(event_generator(), ping=15)


@board_router.post("", response_model=BoardGroupMemoryRead)
async def create_board_group_memory_for_board(
    payload: BoardGroupMemoryCreate,
    board: Board = Depends(get_board_or_404),
    session: AsyncSession = Depends(get_session),
    actor: ActorContext = Depends(require_admin_or_agent),
) -> BoardGroupMemory:
    if actor.actor_type == "agent" and actor.agent:
        if actor.agent.board_id and actor.agent.board_id != board.id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
    group_id = board.board_group_id
    if group_id is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Board is not in a board group",
        )
    group = await session.get(BoardGroup, group_id)
    if group is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)

    tags = set(payload.tags or [])
    is_chat = "chat" in tags
    mentions = extract_mentions(payload.content)
    should_notify = is_chat or "broadcast" in tags or "all" in mentions
    source = payload.source
    if should_notify and not source:
        if actor.actor_type == "agent" and actor.agent:
            source = actor.agent.name
        elif actor.user:
            source = actor.user.preferred_name or actor.user.name or "User"
    memory = BoardGroupMemory(
        board_group_id=group_id,
        content=payload.content,
        tags=payload.tags,
        is_chat=is_chat,
        source=source,
    )
    session.add(memory)
    await session.commit()
    await session.refresh(memory)
    if should_notify:
        await _notify_group_memory_targets(session=session, group=group, memory=memory, actor=actor)
    return memory


router.include_router(group_router)
router.include_router(board_router)
