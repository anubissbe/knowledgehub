"""Create memory sessions table

Revision ID: memory_sessions_001
Revises: 
Create Date: 2025-07-07 19:30:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'memory_sessions_001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    """Create sessions table for memory system"""
    op.create_table(
        'memory_sessions',
        # Primary key
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, 
                  server_default=sa.text('gen_random_uuid()'),
                  nullable=False),
        
        # User and project identification
        sa.Column('user_id', sa.String(255), nullable=False, 
                  comment='Unique identifier for the user'),
        sa.Column('project_id', postgresql.UUID(as_uuid=True), nullable=True,
                  comment='Optional project association'),
        
        # Session lifecycle
        sa.Column('started_at', sa.TIMESTAMP(timezone=True), nullable=False,
                  server_default=sa.text('CURRENT_TIMESTAMP'),
                  comment='When the session started'),
        sa.Column('ended_at', sa.TIMESTAMP(timezone=True), nullable=True,
                  comment='When the session ended'),
        
        # Session metadata
        sa.Column('metadata', postgresql.JSONB, nullable=True,
                  server_default=sa.text("'{}'::jsonb"),
                  comment='Flexible metadata storage'),
        sa.Column('tags', postgresql.ARRAY(sa.Text), nullable=True,
                  server_default=sa.text("'{}'::text[]"),
                  comment='Session tags for categorization'),
        
        # Session linking
        sa.Column('parent_session_id', postgresql.UUID(as_uuid=True), nullable=True,
                  comment='Link to parent session for continuity'),
        
        # Tracking fields
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), nullable=False,
                  server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), nullable=False,
                  server_default=sa.text('CURRENT_TIMESTAMP')),
        
        # Constraints
        sa.ForeignKeyConstraint(['parent_session_id'], ['memory_sessions.id'],
                                name='fk_sessions_parent',
                                ondelete='SET NULL'),
        sa.CheckConstraint('ended_at IS NULL OR ended_at >= started_at',
                           name='ck_sessions_valid_duration'),
        
        # Table configuration
        comment='Stores Claude-Code conversation sessions for memory continuity'
    )
    
    # Create indexes for performance
    op.create_index('idx_memory_sessions_user_id', 'memory_sessions', ['user_id'])
    op.create_index('idx_memory_sessions_project_id', 'memory_sessions', ['project_id'])
    op.create_index('idx_memory_sessions_started_at', 'memory_sessions', ['started_at'])
    op.create_index('idx_memory_sessions_parent_id', 'memory_sessions', ['parent_session_id'])
    op.create_index('idx_memory_sessions_tags', 'memory_sessions', ['tags'], 
                    postgresql_using='gin')
    op.create_index('idx_memory_sessions_metadata', 'memory_sessions', ['metadata'],
                    postgresql_using='gin')
    
    # Create composite index for common queries
    op.create_index('idx_memory_sessions_user_project_time', 'memory_sessions',
                    ['user_id', 'project_id', 'started_at'])
    
    # Create trigger for updated_at
    op.execute("""
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $$ language 'plpgsql';
        
        CREATE TRIGGER update_memory_sessions_updated_at 
        BEFORE UPDATE ON memory_sessions
        FOR EACH ROW 
        EXECUTE FUNCTION update_updated_at_column();
    """)


def downgrade():
    """Drop sessions table and related objects"""
    # Drop trigger first
    op.execute("DROP TRIGGER IF EXISTS update_memory_sessions_updated_at ON memory_sessions")
    op.execute("DROP FUNCTION IF EXISTS update_updated_at_column()")
    
    # Drop indexes
    op.drop_index('idx_memory_sessions_user_project_time')
    op.drop_index('idx_memory_sessions_metadata')
    op.drop_index('idx_memory_sessions_tags')
    op.drop_index('idx_memory_sessions_parent_id')
    op.drop_index('idx_memory_sessions_started_at')
    op.drop_index('idx_memory_sessions_project_id')
    op.drop_index('idx_memory_sessions_user_id')
    
    # Drop table
    op.drop_table('memory_sessions')