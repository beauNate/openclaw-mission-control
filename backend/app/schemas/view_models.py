from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlmodel import Field, SQLModel

from app.schemas.agents import AgentRead
from app.schemas.approvals import ApprovalRead
from app.schemas.board_groups import BoardGroupRead
from app.schemas.board_memory import BoardMemoryRead
from app.schemas.boards import BoardRead
from app.schemas.tasks import TaskRead


class TaskCardRead(TaskRead):
    assignee: str | None = None
    approvals_count: int = 0
    approvals_pending_count: int = 0


class BoardSnapshot(SQLModel):
    board: BoardRead
    tasks: list[TaskCardRead]
    agents: list[AgentRead]
    approvals: list[ApprovalRead]
    chat_messages: list[BoardMemoryRead]
    pending_approvals_count: int = 0


class BoardGroupTaskSummary(SQLModel):
    id: UUID
    board_id: UUID
    board_name: str
    title: str
    status: str
    priority: str
    assigned_agent_id: UUID | None = None
    assignee: str | None = None
    due_at: datetime | None = None
    in_progress_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class BoardGroupBoardSnapshot(SQLModel):
    board: BoardRead
    task_counts: dict[str, int] = Field(default_factory=dict)
    tasks: list[BoardGroupTaskSummary] = Field(default_factory=list)


class BoardGroupSnapshot(SQLModel):
    group: BoardGroupRead | None = None
    boards: list[BoardGroupBoardSnapshot] = Field(default_factory=list)
