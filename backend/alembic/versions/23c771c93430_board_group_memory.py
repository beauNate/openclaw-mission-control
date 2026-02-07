"""board group memory

Revision ID: 23c771c93430
Revises: 12772fdcdfe9
Create Date: 2026-02-07 18:00:19.065861

"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
import sqlmodel


# revision identifiers, used by Alembic.
revision = "23c771c93430"
down_revision = "12772fdcdfe9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Repair drift: it's possible to end up with alembic_version stamped at 12772fdcdfe9
    # without actually applying the board groups schema changes. This migration makes the
    # required board_groups + boards.board_group_id objects exist before adding group memory.
    conn = op.get_bind()
    inspector = sa.inspect(conn)

    if not inspector.has_table("board_groups"):
        op.create_table(
            "board_groups",
            sa.Column("id", sa.Uuid(), nullable=False),
            sa.Column("name", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
            sa.Column("slug", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
            sa.Column("description", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_board_groups_slug", "board_groups", ["slug"], unique=False)
    else:
        indexes = {idx.get("name") for idx in inspector.get_indexes("board_groups")}
        if "ix_board_groups_slug" not in indexes:
            op.create_index("ix_board_groups_slug", "board_groups", ["slug"], unique=False)

    inspector = sa.inspect(conn)
    board_cols = {col.get("name") for col in inspector.get_columns("boards")}
    if "board_group_id" not in board_cols:
        op.add_column("boards", sa.Column("board_group_id", sa.Uuid(), nullable=True))

    inspector = sa.inspect(conn)
    board_indexes = {idx.get("name") for idx in inspector.get_indexes("boards")}
    if "ix_boards_board_group_id" not in board_indexes:
        op.create_index("ix_boards_board_group_id", "boards", ["board_group_id"], unique=False)

    def _has_board_groups_fk() -> bool:
        for fk in inspector.get_foreign_keys("boards"):
            if fk.get("referred_table") != "board_groups":
                continue
            if fk.get("constrained_columns") != ["board_group_id"]:
                continue
            if fk.get("referred_columns") != ["id"]:
                continue
            return True
        return False

    if not _has_board_groups_fk():
        op.create_foreign_key(
            "fk_boards_board_group_id_board_groups",
            "boards",
            "board_groups",
            ["board_group_id"],
            ["id"],
        )

    inspector = sa.inspect(conn)
    if not inspector.has_table("board_group_memory"):
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
    op.drop_index(
        "ix_board_group_memory_board_group_id_is_chat_created_at",
        table_name="board_group_memory",
    )
    op.drop_index("ix_board_group_memory_is_chat", table_name="board_group_memory")
    op.drop_index("ix_board_group_memory_board_group_id", table_name="board_group_memory")
    op.drop_table("board_group_memory")
