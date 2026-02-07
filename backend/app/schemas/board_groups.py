from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlmodel import SQLModel


class BoardGroupBase(SQLModel):
    name: str
    slug: str
    description: str | None = None


class BoardGroupCreate(BoardGroupBase):
    pass


class BoardGroupUpdate(SQLModel):
    name: str | None = None
    slug: str | None = None
    description: str | None = None


class BoardGroupRead(BoardGroupBase):
    id: UUID
    created_at: datetime
    updated_at: datetime
