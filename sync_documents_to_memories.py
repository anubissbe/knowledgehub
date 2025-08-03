#!/usr/bin/env python3
"""
Sync KnowledgeHub documents to memory items for AI Intelligence features
"""

import os
import sys
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import hashlib
import json

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://knowledgehub:knowledgehub@localhost:5433/knowledgehub")

# Create database engine
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

def calculate_content_hash(content):
    """Calculate SHA256 hash of content"""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def sync_documents_to_memories():
    """Sync all documents to memory_items table"""
    db = SessionLocal()
    
    try:
        # Count existing memories
        existing_count = db.execute(text("SELECT COUNT(*) FROM memory_items")).scalar()
        print(f"Existing memories: {existing_count}")
        
        # Count documents
        doc_count = db.execute(text("SELECT COUNT(*) FROM documents")).scalar()
        print(f"Total documents: {doc_count}")
        
        # Get all documents first, we'll filter in Python
        query = text("""
            SELECT d.id, d.title, d.content, d.url, d.created_at, d.source_id
            FROM documents d
            ORDER BY d.created_at DESC
            -- No limit, sync all documents
        """)
        
        documents = db.execute(query).fetchall()
        
        # Get existing content hashes
        existing_hashes = set()
        hash_query = text("SELECT content_hash FROM memory_items WHERE content_hash IS NOT NULL")
        for row in db.execute(hash_query):
            existing_hashes.add(row.content_hash)
        
        print(f"Documents fetched: {len(documents)}")
        print(f"Existing memory hashes: {len(existing_hashes)}")
        
        synced = 0
        errors = 0
        skipped = 0
        
        for doc in documents:
            try:
                # Create memory item from document
                content = f"Document: {doc.title or 'Untitled'}\n\n{doc.content or ''}"
                content_hash = calculate_content_hash(content)
                
                # Skip if already exists
                if content_hash in existing_hashes:
                    skipped += 1
                    continue
                
                # Build tags
                tags = ["document", f"source:{doc.source_id}"]
                if doc.url:
                    domain = doc.url.split('/')[2] if '/' in doc.url else doc.url
                    tags.append(f"domain:{domain}")
                
                # Build metadata
                metadata = {
                    "document_id": str(doc.id),
                    "original_url": doc.url,
                    "title": doc.title,
                    "source_id": str(doc.source_id),
                    "synced_at": datetime.utcnow().isoformat()
                }
                
                # Insert memory item
                insert_query = text("""
                    INSERT INTO memory_items (
                        content, content_hash, tags, metadata, created_at, updated_at, accessed_at
                    ) VALUES (
                        :content, :content_hash, :tags, :metadata, :created_at, NOW(), NOW()
                    )
                    ON CONFLICT (content_hash) DO NOTHING
                """)
                
                result = db.execute(insert_query, {
                    "content": content,
                    "content_hash": content_hash,
                    "tags": tags,
                    "metadata": json.dumps(metadata),
                    "created_at": doc.created_at
                })
                
                if result.rowcount > 0:
                    synced += 1
                    if synced % 100 == 0:
                        db.commit()
                        print(f"Synced {synced} documents...")
                        
            except Exception as e:
                errors += 1
                print(f"Error syncing document {doc.id}: {e}")
                continue
        
        # Final commit
        db.commit()
        
        # Print summary
        print(f"\nSync completed!")
        print(f"Successfully synced: {synced}")
        print(f"Skipped (already exists): {skipped}")
        print(f"Errors: {errors}")
        
        # Verify final count
        final_count = db.execute(text("SELECT COUNT(*) FROM memory_items")).scalar()
        print(f"Total memories after sync: {final_count}")
        
        # Show memory type breakdown
        type_query = text("""
            SELECT tags, COUNT(*) as count
            FROM memory_items
            GROUP BY tags
            ORDER BY count DESC
            LIMIT 10
        """)
        
        print("\nMemory tag distribution:")
        for row in db.execute(type_query):
            print(f"  {row.tags}: {row.count}")
            
    except Exception as e:
        print(f"Fatal error: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    print("Starting document to memory sync...")
    sync_documents_to_memories()