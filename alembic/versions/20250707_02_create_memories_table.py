"""Create memories table

Revision ID: memories_001
Revises: memory_sessions_001
Create Date: 2025-07-07 19:35:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'memories_001'
down_revision = 'memory_sessions_001'
branch_labels = None
depends_on = None


def upgrade():
    """Create memories table for storing extracted information"""
    
    # Create enum type for memory types
    memory_type_enum = postgresql.ENUM(
        'fact', 'preference', 'code', 'decision', 'error', 'pattern', 'entity',
        name='memory_type',
        create_type=True
    )
    memory_type_enum.create(op.get_bind(), checkfirst=True)
    
    op.create_table(
        'memories',
        # Primary key
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text('gen_random_uuid()'),
                  nullable=False),
        
        # Session association
        sa.Column('session_id', postgresql.UUID(as_uuid=True), nullable=False,
                  comment='Session this memory belongs to'),
        
        # Memory content
        sa.Column('content', sa.Text, nullable=False,
                  comment='The actual memory content'),
        sa.Column('summary', sa.Text, nullable=True,
                  comment='Condensed version for quick reference'),
        
        # Memory classification
        sa.Column('memory_type', sa.Enum('fact', 'preference', 'code', 'decision', 
                                         'error', 'pattern', 'entity',
                                         name='memory_type'),
                  nullable=False,
                  comment='Type of memory for categorization'),
        
        # Importance and relevance
        sa.Column('importance', sa.Float, nullable=False, default=0.5,
                  comment='Importance score from 0.0 to 1.0'),
        sa.Column('confidence', sa.Float, nullable=False, default=0.8,
                  comment='Confidence in extraction accuracy'),
        
        # Entity and relationship tracking
        sa.Column('entities', postgresql.ARRAY(sa.Text), nullable=True,
                  server_default=sa.text("'{}'::text[]"),
                  comment='Entities mentioned in this memory'),
        sa.Column('related_memories', postgresql.ARRAY(postgresql.UUID(as_uuid=True)),
                  nullable=True,
                  server_default=sa.text("'{}'::uuid[]"),
                  comment='UUIDs of related memories'),
        
        # Vector embedding for similarity search
        sa.Column('embedding', postgresql.ARRAY(sa.Float), nullable=True,
                  comment='Vector embedding for semantic search'),
        
        # Access tracking
        sa.Column('access_count', sa.Integer, nullable=False, default=0,
                  comment='Number of times this memory was accessed'),
        sa.Column('last_accessed', sa.TIMESTAMP(timezone=True), nullable=True,
                  comment='Last time this memory was accessed'),
        
        # Metadata
        sa.Column('metadata', postgresql.JSONB, nullable=True,
                  server_default=sa.text("'{}'::jsonb"),
                  comment='Additional flexible metadata'),
        
        # Timestamps
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), nullable=False,
                  server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), nullable=False,
                  server_default=sa.text('CURRENT_TIMESTAMP')),
        
        # Constraints
        sa.ForeignKeyConstraint(['session_id'], ['memory_sessions.id'],
                                name='fk_memories_session',
                                ondelete='CASCADE'),
        sa.CheckConstraint('importance >= 0 AND importance <= 1',
                           name='ck_memories_importance_range'),
        sa.CheckConstraint('confidence >= 0 AND confidence <= 1',
                           name='ck_memories_confidence_range'),
        sa.CheckConstraint('access_count >= 0',
                           name='ck_memories_access_count_positive'),
        
        # Table configuration
        comment='Stores extracted memories from Claude-Code conversations'
    )
    
    # Create indexes for performance
    op.create_index('idx_memories_session_id', 'memories', ['session_id'])
    op.create_index('idx_memories_type', 'memories', ['memory_type'])
    op.create_index('idx_memories_importance', 'memories', ['importance'], 
                    postgresql_order='DESC')
    op.create_index('idx_memories_entities', 'memories', ['entities'],
                    postgresql_using='gin')
    op.create_index('idx_memories_metadata', 'memories', ['metadata'],
                    postgresql_using='gin')
    op.create_index('idx_memories_created_at', 'memories', ['created_at'])
    op.create_index('idx_memories_last_accessed', 'memories', ['last_accessed'])
    
    # Composite indexes for common queries
    op.create_index('idx_memories_session_type_importance', 'memories',
                    ['session_id', 'memory_type', 'importance'])
    op.create_index('idx_memories_type_importance_created', 'memories',
                    ['memory_type', 'importance', 'created_at'])
    
    # Create trigger for updated_at
    op.execute("""
        CREATE TRIGGER update_memories_updated_at 
        BEFORE UPDATE ON memories
        FOR EACH ROW 
        EXECUTE FUNCTION update_updated_at_column();
    """)
    
    # Create function for updating access tracking
    op.execute("""
        CREATE OR REPLACE FUNCTION update_memory_access()
        RETURNS TRIGGER AS $$
        BEGIN
            IF TG_OP = 'SELECT' THEN
                UPDATE memories 
                SET access_count = access_count + 1,
                    last_accessed = CURRENT_TIMESTAMP
                WHERE id = NEW.id;
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)


def downgrade():
    """Drop memories table and related objects"""
    # Drop triggers and functions
    op.execute("DROP TRIGGER IF EXISTS update_memories_updated_at ON memories")
    op.execute("DROP FUNCTION IF EXISTS update_memory_access()")
    
    # Drop indexes
    op.drop_index('idx_memories_type_importance_created')
    op.drop_index('idx_memories_session_type_importance')
    op.drop_index('idx_memories_last_accessed')
    op.drop_index('idx_memories_created_at')
    op.drop_index('idx_memories_metadata')
    op.drop_index('idx_memories_entities')
    op.drop_index('idx_memories_importance')
    op.drop_index('idx_memories_type')
    op.drop_index('idx_memories_session_id')
    
    # Drop table
    op.drop_table('memories')
    
    # Drop enum type
    op.execute("DROP TYPE IF EXISTS memory_type")