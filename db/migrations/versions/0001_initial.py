"""initial: pgvector + pending_approvals + code_runs

Revision ID: 0001_initial
Revises:
Create Date: 2026-06-21
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None

EMBED_DIM = 768


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "pending_approvals",
        sa.Column("request_id", sa.String(64), primary_key=True),
        sa.Column("action_type", sa.String(64), nullable=False),
        sa.Column("conversation_id", sa.String(500), nullable=False),
        sa.Column("payload_json", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("payload_sha256", sa.String(64), nullable=False, server_default=""),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("requested_by", sa.String(320), nullable=False, server_default=""),
        sa.Column("approver_email", sa.String(320), nullable=True),
        sa.Column("activity_id", sa.String(200), nullable=True),
        sa.Column("revise_note", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_pending_approvals_status", "pending_approvals", ["status"])

    op.create_table(
        "code_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("language", sa.String(32), nullable=False, server_default="python"),
        sa.Column("script", sa.Text(), nullable=False),
        sa.Column("inputs_hash", sa.String(64), nullable=False, server_default=""),
        sa.Column("outcome", sa.String(16), nullable=False, server_default="ok"),
        sa.Column("result_summary", sa.Text(), nullable=False, server_default=""),
        sa.Column("approved_by", sa.String(320), nullable=True),
        sa.Column("reusable", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("embedding", Vector(EMBED_DIM), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index("ix_code_runs_reusable", "code_runs", ["reusable"])


def downgrade() -> None:
    op.drop_index("ix_code_runs_reusable", table_name="code_runs")
    op.drop_table("code_runs")
    op.drop_index("ix_pending_approvals_status", table_name="pending_approvals")
    op.drop_table("pending_approvals")
    # Leave the `vector` extension installed (other objects may use it).
