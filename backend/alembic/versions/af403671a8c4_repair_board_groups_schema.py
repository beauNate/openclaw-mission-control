"""repair board groups schema

Revision ID: af403671a8c4
Revises: 5fb3b2491090
Create Date: 2026-02-07

"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
import sqlmodel

# revision identifiers, used by Alembic.
revision = "af403671a8c4"
down_revision = "5fb3b2491090"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Repair drift: it is possible to end up with alembic_version stamped at (or beyond)
    # the board group revisions without having the underlying DB objects present.
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


def downgrade() -> None:
    # Repair migration: do not attempt to undo drift fixes automatically.
    return

