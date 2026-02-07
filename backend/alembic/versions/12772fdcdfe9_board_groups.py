"""board groups

Revision ID: 12772fdcdfe9
Revises: 9f0c4fb2a7b8
Create Date: 2026-02-07 17:13:50.597099

"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
import sqlmodel


# revision identifiers, used by Alembic.
revision = "12772fdcdfe9"
down_revision = "9f0c4fb2a7b8"
branch_labels = None
depends_on = None


def upgrade() -> None:
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

    op.add_column("boards", sa.Column("board_group_id", sa.Uuid(), nullable=True))
    op.create_index("ix_boards_board_group_id", "boards", ["board_group_id"], unique=False)
    op.create_foreign_key(
        "fk_boards_board_group_id_board_groups",
        "boards",
        "board_groups",
        ["board_group_id"],
        ["id"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "fk_boards_board_group_id_board_groups", "boards", type_="foreignkey"
    )
    op.drop_index("ix_boards_board_group_id", table_name="boards")
    op.drop_column("boards", "board_group_id")

    op.drop_index("ix_board_groups_slug", table_name="board_groups")
    op.drop_table("board_groups")
