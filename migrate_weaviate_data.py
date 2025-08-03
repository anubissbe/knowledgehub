#!/usr/bin/env python3
"""
Migrate data from KnowledgeChunk to Knowledge_chunks collection
and ensure chunk_type values are lowercase
"""

import weaviate
import logging
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate_weaviate_data():
    """Migrate data between Weaviate collections"""
    
    # Connect to Weaviate
    client = weaviate.Client(url="http://localhost:8090")
    
    # Check both collections exist
    schema = client.schema.get()
    collections = [c['class'] for c in schema.get('classes', [])]
    
    if 'KnowledgeChunk' not in collections:
        logger.error("Source collection 'KnowledgeChunk' not found")
        return
        
    if 'Knowledge_chunks' not in collections:
        logger.error("Target collection 'Knowledge_chunks' not found")
        return
    
    # Get all data from KnowledgeChunk
    logger.info("Fetching data from KnowledgeChunk collection...")
    
    # Query in batches
    batch_size = 100
    offset = 0
    total_migrated = 0
    
    while True:
        # Fetch batch
        result = client.query.get(
            'KnowledgeChunk',
            ['content', 'title', 'url', 'source_id', 'document_id', 
             'chunk_index', 'source_type', 'metadata', 'created_at']
        ).with_additional(['id', 'vector']).with_limit(batch_size).with_offset(offset).do()
        
        if 'errors' in result:
            logger.error(f"Query error: {result['errors']}")
            break
            
        chunks = result.get('data', {}).get('Get', {}).get('KnowledgeChunk', [])
        
        if not chunks:
            logger.info("No more chunks to migrate")
            break
            
        logger.info(f"Processing batch of {len(chunks)} chunks (offset: {offset})")
        
        # Prepare batch for insertion
        with client.batch as batch:
            batch.batch_size = 100
            
            for chunk in chunks:
                # Map fields to new schema
                data_object = {
                    'chunk_index': chunk.get('chunk_index', 0),
                    'doc_id': chunk.get('document_id', ''),
                    'source_id': chunk.get('source_id', ''),
                    'content': chunk.get('content', ''),
                    'chunk_type': 'text',  # Default to 'text' (lowercase)
                    'position': chunk.get('chunk_index', 0),
                    'metadata': chunk.get('metadata', '{}'),
                    'created_at': chunk.get('created_at', datetime.utcnow().isoformat())
                }
                
                # Get vector if available
                vector = None
                if '_additional' in chunk and 'vector' in chunk['_additional']:
                    vector = chunk['_additional']['vector']
                
                # Add to batch
                batch.add_data_object(
                    data_object=data_object,
                    class_name='Knowledge_chunks',
                    vector=vector
                )
            
            # Execute batch
            batch.create_objects()
            
        total_migrated += len(chunks)
        offset += batch_size
        
    logger.info(f"Migration complete! Total chunks migrated: {total_migrated}")
    
    # Verify migration
    verify_result = client.query.aggregate('Knowledge_chunks').with_meta_count().do()
    count = verify_result['data']['Aggregate']['Knowledge_chunks'][0]['meta']['count']
    logger.info(f"Verification: Knowledge_chunks now contains {count} objects")
    
    # Optional: Delete old collection
    response = input("\nDo you want to delete the old 'KnowledgeChunk' collection? (yes/no): ")
    if response.lower() == 'yes':
        client.schema.delete_class('KnowledgeChunk')
        logger.info("Deleted old KnowledgeChunk collection")
    else:
        logger.info("Keeping old KnowledgeChunk collection")

if __name__ == "__main__":
    migrate_weaviate_data()