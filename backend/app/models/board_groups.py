from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlmodel import Field

from app.core.time import utcnow
from app.models.tenancy import TenantScoped


class BoardGroup(TenantScoped, table=True):
    __tablename__ = "board_groups"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    name: str
    slug: str = Field(index=True)
    description: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
