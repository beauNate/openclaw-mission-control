from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlmodel import SQLModel

from app.schemas.common import NonEmptyStr


class BoardGroupMemoryCreate(SQLModel):
    # For writes, reject blank/whitespace-only content.
    content: NonEmptyStr
    tags: list[str] | None = None
    source: str | None = None


class BoardGroupMemoryRead(SQLModel):
    id: UUID
    board_group_id: UUID
    # For reads, allow legacy rows that may have empty content (avoid response validation 500s).
    content: str
    tags: list[str] | None = None
    source: str | None = None
    is_chat: bool = False
    created_at: datetime
