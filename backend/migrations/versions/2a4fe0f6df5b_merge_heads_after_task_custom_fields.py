"""Merge heads after task custom fields.

Revision ID: 2a4fe0f6df5b
Revises: b6f4c7d9e1a2, d3ca36cf31a1
Create Date: 2026-02-13 21:43:07

"""

from __future__ import annotations


# revision identifiers, used by Alembic.
revision = "2a4fe0f6df5b"
down_revision = ("b6f4c7d9e1a2", "d3ca36cf31a1")
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Merge heads."""
    pass


def downgrade() -> None:
    """Unmerge heads."""
    pass
