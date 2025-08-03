"""
Create chunks for existing Checkmarx documents that have no chunks
"""
import psycopg2
from psycopg2.extras import Json
import uuid
from datetime import datetime
import re

def create_chunks_for_documents():
    # Connect to database
    conn = psycopg2.connect(
        host="localhost",
        port=5433,
        database="knowledgehub",
        user="knowledgehub",
        password="knowledgehub"
    )
    
    cur = conn.cursor()
    
    # Get Checkmarx API Guide source ID
    source_id = 'a2ef8910-0b25-4138-abcb-428666ce691d'
    
    # Get all documents without chunks
    cur.execute("""
        SELECT d.id, d.url, d.title, d.content 
        FROM documents d
        LEFT JOIN document_chunks dc ON d.id = dc.document_id
        WHERE d.source_id = %s 
        AND dc.id IS NULL
        AND d.content IS NOT NULL
        AND LENGTH(d.content) > 0
    """, (source_id,))
    
    documents = cur.fetchall()
    print(f"Found {len(documents)} documents without chunks")
    
    chunks_created = 0
    
    for doc_id, url, title, content in documents:
        if not content or len(content.strip()) < 10:
            continue
            
        # Simple chunking strategy - split by paragraphs or sections
        # For API docs, we'll try to preserve meaningful sections
        
        chunks = []
        
        # Split by double newlines first (paragraphs)
        paragraphs = content.split('\n\n')
        
        current_chunk = ""
        chunk_index = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            # If adding this paragraph would make chunk too large, save current chunk
            if len(current_chunk) > 0 and len(current_chunk) + len(para) > 2000:
                chunks.append({
                    'content': current_chunk.strip(),
                    'index': chunk_index
                })
                chunk_index += 1
                current_chunk = para
            else:
                current_chunk += "\n\n" + para if current_chunk else para
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append({
                'content': current_chunk.strip(),
                'index': chunk_index
            })
        
        # If we didn't get any chunks from paragraph splitting, just chunk by size
        if not chunks and len(content) > 100:
            # Chunk by sentences or fixed size
            words = content.split()
            chunk_size = 500  # words per chunk
            
            for i in range(0, len(words), chunk_size):
                chunk_content = ' '.join(words[i:i+chunk_size])
                if len(chunk_content.strip()) > 10:
                    chunks.append({
                        'content': chunk_content,
                        'index': i // chunk_size
                    })
        
        # Insert chunks into database
        for chunk in chunks:
            chunk_id = str(uuid.uuid4())
            
            # Determine chunk type - for API docs, it's usually "text"
            chunk_type = 'TEXT'  # PostgreSQL enum values are usually uppercase
            
            try:
                cur.execute("""
                    INSERT INTO document_chunks 
                    (id, document_id, chunk_index, chunk_type, content, metadata, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, NOW())
                """, (
                    chunk_id,
                    doc_id,
                    chunk['index'],
                    chunk_type,
                    chunk['content'],
                    Json({
                        'source_url': url,
                        'source_title': title,
                        'chunk_method': 'paragraph_split'
                    })
                ))
                chunks_created += 1
            except Exception as e:
                print(f"Error creating chunk for document {doc_id}: {e}")
                conn.rollback()
                continue
        
        if chunks:
            print(f"Created {len(chunks)} chunks for: {title}")
    
    # Update source stats
    cur.execute("""
        UPDATE knowledge_sources 
        SET stats = jsonb_set(stats, '{chunks}', %s::jsonb)
        WHERE id = %s
    """, (str(chunks_created), source_id))
    
    conn.commit()
    cur.close()
    conn.close()
    
    print(f"\n✅ Successfully created {chunks_created} chunks for Checkmarx documents!")
    
    return chunks_created

if __name__ == "__main__":
    create_chunks_for_documents()