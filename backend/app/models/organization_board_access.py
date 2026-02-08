from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import UniqueConstraint
from sqlmodel import Field, SQLModel

from app.core.time import utcnow


class OrganizationBoardAccess(SQLModel, table=True):
    __tablename__ = "organization_board_access"
    __table_args__ = (
        UniqueConstraint(
            "organization_member_id",
            "board_id",
            name="uq_org_board_access_member_board",
        ),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    organization_member_id: UUID = Field(foreign_key="organization_members.id", index=True)
    board_id: UUID = Field(foreign_key="boards.id", index=True)
    can_read: bool = Field(default=True)
    can_write: bool = Field(default=False)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
