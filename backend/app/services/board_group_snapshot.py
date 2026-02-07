from __future__ import annotations

from collections import defaultdict
from typing import Any
from uuid import UUID

from sqlalchemy import case, func
from sqlmodel import col, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.models.agents import Agent
from app.models.board_groups import BoardGroup
from app.models.boards import Board
from app.models.tasks import Task
from app.schemas.board_groups import BoardGroupRead
from app.schemas.boards import BoardRead
from app.schemas.view_models import (
    BoardGroupBoardSnapshot,
    BoardGroupSnapshot,
    BoardGroupTaskSummary,
)

_STATUS_ORDER = {"in_progress": 0, "review": 1, "inbox": 2, "done": 3}
_PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def _status_weight_expr() -> Any:
    whens = [(col(Task.status) == key, weight) for key, weight in _STATUS_ORDER.items()]
    return case(*whens, else_=99)


def _priority_weight_expr() -> Any:
    whens = [(col(Task.priority) == key, weight) for key, weight in _PRIORITY_ORDER.items()]
    return case(*whens, else_=99)


async def build_group_snapshot(
    session: AsyncSession,
    *,
    group: BoardGroup,
    exclude_board_id: UUID | None = None,
    include_done: bool = False,
    per_board_task_limit: int = 5,
) -> BoardGroupSnapshot:
    statement = select(Board).where(col(Board.board_group_id) == group.id)
    if exclude_board_id is not None:
        statement = statement.where(col(Board.id) != exclude_board_id)
    boards = list(await session.exec(statement.order_by(func.lower(col(Board.name)).asc())))
    if not boards:
        return BoardGroupSnapshot(group=BoardGroupRead.model_validate(group, from_attributes=True))

    boards_by_id = {board.id: board for board in boards}
    board_ids = list(boards_by_id.keys())

    task_counts: dict[UUID, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for board_id, status_value, total in list(
        await session.exec(
            select(col(Task.board_id), col(Task.status), func.count(col(Task.id)))
            .where(col(Task.board_id).in_(board_ids))
            .group_by(col(Task.board_id), col(Task.status))
        )
    ):
        if board_id is None:
            continue
        task_counts[board_id][str(status_value)] = int(total or 0)

    task_statement = select(Task).where(col(Task.board_id).in_(board_ids))
    if not include_done:
        task_statement = task_statement.where(col(Task.status) != "done")
    task_statement = task_statement.order_by(
        col(Task.board_id).asc(),
        _status_weight_expr().asc(),
        _priority_weight_expr().asc(),
        col(Task.updated_at).desc(),
        col(Task.created_at).desc(),
    )
    tasks = list(await session.exec(task_statement))

    assigned_ids = {task.assigned_agent_id for task in tasks if task.assigned_agent_id is not None}
    agent_name_by_id: dict[UUID, str] = {}
    if assigned_ids:
        for agent_id, name in list(
            await session.exec(
                select(col(Agent.id), col(Agent.name)).where(col(Agent.id).in_(assigned_ids))
            )
        ):
            agent_name_by_id[agent_id] = name

    tasks_by_board: dict[UUID, list[BoardGroupTaskSummary]] = defaultdict(list)
    if per_board_task_limit > 0:
        for task in tasks:
            if task.board_id is None:
                continue
            current = tasks_by_board[task.board_id]
            if len(current) >= per_board_task_limit:
                continue
            board = boards_by_id.get(task.board_id)
            if board is None:
                continue
            current.append(
                BoardGroupTaskSummary(
                    id=task.id,
                    board_id=task.board_id,
                    board_name=board.name,
                    title=task.title,
                    status=task.status,
                    priority=task.priority,
                    assigned_agent_id=task.assigned_agent_id,
                    assignee=(
                        agent_name_by_id.get(task.assigned_agent_id)
                        if task.assigned_agent_id is not None
                        else None
                    ),
                    due_at=task.due_at,
                    in_progress_at=task.in_progress_at,
                    created_at=task.created_at,
                    updated_at=task.updated_at,
                )
            )

    snapshots: list[BoardGroupBoardSnapshot] = []
    for board in boards:
        board_read = BoardRead.model_validate(board, from_attributes=True)
        counts = dict(task_counts.get(board.id, {}))
        snapshots.append(
            BoardGroupBoardSnapshot(
                board=board_read,
                task_counts=counts,
                tasks=tasks_by_board.get(board.id, []),
            )
        )

    return BoardGroupSnapshot(
        group=BoardGroupRead.model_validate(group, from_attributes=True),
        boards=snapshots,
    )


async def build_board_group_snapshot(
    session: AsyncSession,
    *,
    board: Board,
    include_self: bool = False,
    include_done: bool = False,
    per_board_task_limit: int = 5,
) -> BoardGroupSnapshot:
    if not board.board_group_id:
        return BoardGroupSnapshot(group=None, boards=[])
    group = await session.get(BoardGroup, board.board_group_id)
    if group is None:
        return BoardGroupSnapshot(group=None, boards=[])
    return await build_group_snapshot(
        session,
        group=group,
        exclude_board_id=None if include_self else board.id,
        include_done=include_done,
        per_board_task_limit=per_board_task_limit,
    )
