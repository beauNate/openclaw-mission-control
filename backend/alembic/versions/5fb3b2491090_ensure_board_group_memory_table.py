"""ensure board group memory table

Revision ID: 5fb3b2491090
Revises: 23c771c93430
Create Date: 2026-02-07 18:07:20.588662

"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "5fb3b2491090"
down_revision = "23c771c93430"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    if inspector.has_table("board_group_memory"):
        return

    op.create_table(
        "board_group_memory",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("board_group_id", sa.Uuid(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("tags", sa.JSON(), nullable=True),
        sa.Column(
            "is_chat",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column("source", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["board_group_id"], ["board_groups.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_board_group_memory_board_group_id",
        "board_group_memory",
        ["board_group_id"],
        unique=False,
    )
    op.create_index(
        "ix_board_group_memory_is_chat",
        "board_group_memory",
        ["is_chat"],
        unique=False,
    )
    op.create_index(
        "ix_board_group_memory_board_group_id_is_chat_created_at",
        "board_group_memory",
        ["board_group_id", "is_chat", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    # This is a repair migration. Downgrading from 5fb3b2491090 -> 23c771c93430
    # should keep the board_group_memory table (it belongs to the prior revision).
    return
