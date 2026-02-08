from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlmodel.ext.asyncio.session import AsyncSession

from app.api.deps import require_org_admin
from app.core.auth import AuthContext, get_auth_context
from app.db.session import get_session
from app.integrations.openclaw_gateway import GatewayConfig as GatewayClientConfig
from app.integrations.openclaw_gateway import (
    OpenClawGatewayError,
    ensure_session,
    get_chat_history,
    openclaw_call,
    send_message,
)
from app.integrations.openclaw_gateway_protocol import (
    GATEWAY_EVENTS,
    GATEWAY_METHODS,
    PROTOCOL_VERSION,
)
from app.models.boards import Board
from app.models.gateways import Gateway
from app.schemas.common import OkResponse
from app.schemas.gateway_api import (
    GatewayCommandsResponse,
    GatewayResolveQuery,
    GatewaySessionHistoryResponse,
    GatewaySessionMessageRequest,
    GatewaySessionResponse,
    GatewaySessionsResponse,
    GatewaysStatusResponse,
)
from app.services.organizations import OrganizationContext, require_board_access

router = APIRouter(prefix="/gateways", tags=["gateways"])


async def _resolve_gateway(
    session: AsyncSession,
    board_id: str | None,
    gateway_url: str | None,
    gateway_token: str | None,
    gateway_main_session_key: str | None,
    *,
    user: object | None = None,
) -> tuple[Board | None, GatewayClientConfig, str | None]:
    if gateway_url:
        return (
            None,
            GatewayClientConfig(url=gateway_url, token=gateway_token),
            gateway_main_session_key,
        )
    if not board_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="board_id or gateway_url is required",
        )
    board = await session.get(Board, board_id)
    if board is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Board not found")
    if isinstance(user, object) and user is not None:
        await require_board_access(session, user=user, board=board, write=False)  # type: ignore[arg-type]
    if not board.gateway_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Board gateway_id is required",
        )
    gateway = await session.get(Gateway, board.gateway_id)
    if gateway is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Board gateway_id is invalid",
        )
    if not gateway.url:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Gateway url is required",
        )
    return (
        board,
        GatewayClientConfig(url=gateway.url, token=gateway.token),
        gateway.main_session_key,
    )


async def _require_gateway(
    session: AsyncSession, board_id: str | None, *, user: object | None = None
) -> tuple[Board, GatewayClientConfig, str | None]:
    board, config, main_session = await _resolve_gateway(
        session,
        board_id,
        None,
        None,
        None,
        user=user,
    )
    if board is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="board_id is required",
        )
    return board, config, main_session


@router.get("/status", response_model=GatewaysStatusResponse)
async def gateways_status(
    params: GatewayResolveQuery = Depends(),
    session: AsyncSession = Depends(get_session),
    auth: AuthContext = Depends(get_auth_context),
    ctx: OrganizationContext = Depends(require_org_admin),
) -> GatewaysStatusResponse:
    board, config, main_session = await _resolve_gateway(
        session,
        params.board_id,
        params.gateway_url,
        params.gateway_token,
        params.gateway_main_session_key,
        user=auth.user,
    )
    if board is not None and board.organization_id != ctx.organization.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
    try:
        sessions = await openclaw_call("sessions.list", config=config)
        if isinstance(sessions, dict):
            sessions_list = list(sessions.get("sessions") or [])
        else:
            sessions_list = list(sessions or [])
        main_session_entry: object | None = None
        main_session_error: str | None = None
        if main_session:
            try:
                ensured = await ensure_session(main_session, config=config, label="Main Agent")
                if isinstance(ensured, dict):
                    main_session_entry = ensured.get("entry") or ensured
            except OpenClawGatewayError as exc:
                main_session_error = str(exc)
        return GatewaysStatusResponse(
            connected=True,
            gateway_url=config.url,
            sessions_count=len(sessions_list),
            sessions=sessions_list,
            main_session_key=main_session,
            main_session=main_session_entry,
            main_session_error=main_session_error,
        )
    except OpenClawGatewayError as exc:
        return GatewaysStatusResponse(connected=False, gateway_url=config.url, error=str(exc))


@router.get("/sessions", response_model=GatewaySessionsResponse)
async def list_gateway_sessions(
    board_id: str | None = Query(default=None),
    session: AsyncSession = Depends(get_session),
    auth: AuthContext = Depends(get_auth_context),
    ctx: OrganizationContext = Depends(require_org_admin),
) -> GatewaySessionsResponse:
    board, config, main_session = await _resolve_gateway(
        session,
        board_id,
        None,
        None,
        None,
        user=auth.user,
    )
    if board is not None and board.organization_id != ctx.organization.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
    try:
        sessions = await openclaw_call("sessions.list", config=config)
    except OpenClawGatewayError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    if isinstance(sessions, dict):
        sessions_list = list(sessions.get("sessions") or [])
    else:
        sessions_list = list(sessions or [])

    main_session_entry: object | None = None
    if main_session:
        try:
            ensured = await ensure_session(main_session, config=config, label="Main Agent")
            if isinstance(ensured, dict):
                main_session_entry = ensured.get("entry") or ensured
        except OpenClawGatewayError:
            main_session_entry = None

    return GatewaySessionsResponse(
        sessions=sessions_list,
        main_session_key=main_session,
        main_session=main_session_entry,
    )


@router.get("/sessions/{session_id}", response_model=GatewaySessionResponse)
async def get_gateway_session(
    session_id: str,
    board_id: str | None = Query(default=None),
    session: AsyncSession = Depends(get_session),
    auth: AuthContext = Depends(get_auth_context),
    ctx: OrganizationContext = Depends(require_org_admin),
) -> GatewaySessionResponse:
    board, config, main_session = await _resolve_gateway(
        session,
        board_id,
        None,
        None,
        None,
        user=auth.user,
    )
    if board is not None and board.organization_id != ctx.organization.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
    try:
        sessions = await openclaw_call("sessions.list", config=config)
    except OpenClawGatewayError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    if isinstance(sessions, dict):
        sessions_list = list(sessions.get("sessions") or [])
    else:
        sessions_list = list(sessions or [])
    if main_session and not any(session.get("key") == main_session for session in sessions_list):
        try:
            await ensure_session(main_session, config=config, label="Main Agent")
            refreshed = await openclaw_call("sessions.list", config=config)
            if isinstance(refreshed, dict):
                sessions_list = list(refreshed.get("sessions") or [])
            else:
                sessions_list = list(refreshed or [])
        except OpenClawGatewayError:
            pass
    session_entry = next((item for item in sessions_list if item.get("key") == session_id), None)
    if session_entry is None and main_session and session_id == main_session:
        try:
            ensured = await ensure_session(main_session, config=config, label="Main Agent")
            if isinstance(ensured, dict):
                session_entry = ensured.get("entry") or ensured
        except OpenClawGatewayError:
            session_entry = None
    if session_entry is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return GatewaySessionResponse(session=session_entry)


@router.get("/sessions/{session_id}/history", response_model=GatewaySessionHistoryResponse)
async def get_session_history(
    session_id: str,
    board_id: str | None = Query(default=None),
    session: AsyncSession = Depends(get_session),
    auth: AuthContext = Depends(get_auth_context),
    ctx: OrganizationContext = Depends(require_org_admin),
) -> GatewaySessionHistoryResponse:
    board, config, _ = await _require_gateway(session, board_id, user=auth.user)
    if board.organization_id != ctx.organization.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
    try:
        history = await get_chat_history(session_id, config=config)
    except OpenClawGatewayError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    if isinstance(history, dict) and isinstance(history.get("messages"), list):
        return GatewaySessionHistoryResponse(history=history["messages"])
    return GatewaySessionHistoryResponse(history=list(history or []))


@router.post("/sessions/{session_id}/message", response_model=OkResponse)
async def send_gateway_session_message(
    session_id: str,
    payload: GatewaySessionMessageRequest,
    board_id: str | None = Query(default=None),
    session: AsyncSession = Depends(get_session),
    auth: AuthContext = Depends(get_auth_context),
    ctx: OrganizationContext = Depends(require_org_admin),
) -> OkResponse:
    board, config, main_session = await _require_gateway(session, board_id, user=auth.user)
    if board.organization_id != ctx.organization.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
    if auth.user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    await require_board_access(session, user=auth.user, board=board, write=True)
    try:
        if main_session and session_id == main_session:
            await ensure_session(main_session, config=config, label="Main Agent")
        await send_message(payload.content, session_key=session_id, config=config)
    except OpenClawGatewayError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    return OkResponse()


@router.get("/commands", response_model=GatewayCommandsResponse)
async def gateway_commands(
    auth: AuthContext = Depends(get_auth_context),
    _ctx: OrganizationContext = Depends(require_org_admin),
) -> GatewayCommandsResponse:
    return GatewayCommandsResponse(
        protocol_version=PROTOCOL_VERSION,
        methods=GATEWAY_METHODS,
        events=GATEWAY_EVENTS,
    )
