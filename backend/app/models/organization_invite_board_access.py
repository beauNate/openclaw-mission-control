from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import UniqueConstraint
from sqlmodel import Field, SQLModel

from app.core.time import utcnow


class OrganizationInviteBoardAccess(SQLModel, table=True):
    __tablename__ = "organization_invite_board_access"
    __table_args__ = (
        UniqueConstraint(
            "organization_invite_id",
            "board_id",
            name="uq_org_invite_board_access_invite_board",
        ),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    organization_invite_id: UUID = Field(foreign_key="organization_invites.id", index=True)
    board_id: UUID = Field(foreign_key="boards.id", index=True)
    can_read: bool = Field(default=True)
    can_write: bool = Field(default=False)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
