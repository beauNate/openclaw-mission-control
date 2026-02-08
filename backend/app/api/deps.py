from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from fastapi import Depends, HTTPException, status
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.agent_auth import AgentAuthContext, get_agent_auth_context_optional
from app.core.auth import AuthContext, get_auth_context, get_auth_context_optional
from app.db.session import get_session
from app.models.agents import Agent
from app.models.boards import Board
from app.models.organizations import Organization
from app.models.tasks import Task
from app.models.users import User
from app.services.admin_access import require_admin
from app.services.organizations import (
    OrganizationContext,
    ensure_member_for_user,
    get_active_membership,
    is_org_admin,
    require_board_access,
)


def require_admin_auth(auth: AuthContext = Depends(get_auth_context)) -> AuthContext:
    require_admin(auth)
    return auth


@dataclass
class ActorContext:
    actor_type: Literal["user", "agent"]
    user: User | None = None
    agent: Agent | None = None


def require_admin_or_agent(
    auth: AuthContext | None = Depends(get_auth_context_optional),
    agent_auth: AgentAuthContext | None = Depends(get_agent_auth_context_optional),
) -> ActorContext:
    if auth is not None:
        require_admin(auth)
        return ActorContext(actor_type="user", user=auth.user)
    if agent_auth is not None:
        return ActorContext(actor_type="agent", agent=agent_auth.agent)
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)


async def require_org_member(
    auth: AuthContext = Depends(get_auth_context),
    session: AsyncSession = Depends(get_session),
) -> OrganizationContext:
    if auth.user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    member = await get_active_membership(session, auth.user)
    if member is None:
        member = await ensure_member_for_user(session, auth.user)
    if member is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
    organization = await session.get(Organization, member.organization_id)
    if organization is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
    return OrganizationContext(organization=organization, member=member)


async def require_org_admin(
    ctx: OrganizationContext = Depends(require_org_member),
) -> OrganizationContext:
    if not is_org_admin(ctx.member):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
    return ctx


async def get_board_or_404(
    board_id: str,
    session: AsyncSession = Depends(get_session),
) -> Board:
    board = await session.get(Board, board_id)
    if board is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    return board


async def get_board_for_actor_read(
    board_id: str,
    session: AsyncSession = Depends(get_session),
    actor: ActorContext = Depends(require_admin_or_agent),
) -> Board:
    board = await session.get(Board, board_id)
    if board is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    if actor.actor_type == "agent":
        if actor.agent and actor.agent.board_id and actor.agent.board_id != board.id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
        return board
    if actor.user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    await require_board_access(session, user=actor.user, board=board, write=False)
    return board


async def get_board_for_actor_write(
    board_id: str,
    session: AsyncSession = Depends(get_session),
    actor: ActorContext = Depends(require_admin_or_agent),
) -> Board:
    board = await session.get(Board, board_id)
    if board is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    if actor.actor_type == "agent":
        if actor.agent and actor.agent.board_id and actor.agent.board_id != board.id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
        return board
    if actor.user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    await require_board_access(session, user=actor.user, board=board, write=True)
    return board


async def get_board_for_user_read(
    board_id: str,
    session: AsyncSession = Depends(get_session),
    auth: AuthContext = Depends(get_auth_context),
) -> Board:
    board = await session.get(Board, board_id)
    if board is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    if auth.user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    await require_board_access(session, user=auth.user, board=board, write=False)
    return board


async def get_board_for_user_write(
    board_id: str,
    session: AsyncSession = Depends(get_session),
    auth: AuthContext = Depends(get_auth_context),
) -> Board:
    board = await session.get(Board, board_id)
    if board is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    if auth.user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    await require_board_access(session, user=auth.user, board=board, write=True)
    return board


async def get_task_or_404(
    task_id: str,
    board: Board = Depends(get_board_for_actor_read),
    session: AsyncSession = Depends(get_session),
) -> Task:
    task = await session.get(Task, task_id)
    if task is None or task.board_id != board.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    return task
