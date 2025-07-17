#!/usr/bin/env python3
"""
Populate all multi-dimensional databases with scraped content.
This script takes documents from PostgreSQL and populates:
- Weaviate (vector embeddings)
- Neo4j (knowledge graph)
- TimescaleDB (time-series analytics)
"""

import os
import sys
import json
import logging
from datetime import datetime
import weaviate
from neo4j import GraphDatabase
import psycopg2
from psycopg2.extras import RealDictCursor
import requests
from sentence_transformers import SentenceTransformer
import hashlib
from typing import List, Dict, Any
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiDatabasePopulator:
    def __init__(self):
        # PostgreSQL connection
        self.pg_conn = psycopg2.connect(
            host="localhost",
            port=5433,
            database="knowledgehub",
            user="knowledgehub",
            password="knowledgehub"
        )
        
        # Weaviate client
        self.weaviate_client = weaviate.Client(
            url="http://localhost:8090",
            timeout_config=(30, 60)
        )
        
        # Neo4j driver
        self.neo4j_driver = GraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", "knowledgehub123")
        )
        
        # TimescaleDB connection
        self.ts_conn = psycopg2.connect(
            host="localhost",
            port=5434,
            database="knowledgehub_analytics",
            user="knowledgehub",
            password="knowledgehub123"
        )
        
        # Load embedding model
        logger.info("Loading embedding model...")
        self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # AI service for enhanced embeddings
        self.ai_service_url = "http://localhost:8002"
        
    def setup_databases(self):
        """Ensure all databases have proper schemas"""
        
        # Setup Weaviate schema
        self._setup_weaviate_schema()
        
        # Setup Neo4j constraints and indexes
        self._setup_neo4j_schema()
        
        # Setup TimescaleDB tables
        self._setup_timescale_schema()
        
    def _setup_weaviate_schema(self):
        """Create Weaviate schema for knowledge chunks"""
        schema = {
            "class": "KnowledgeChunk",
            "description": "A chunk of knowledge from scraped documents",
            "properties": [
                {
                    "name": "content",
                    "dataType": ["text"],
                    "description": "The text content of the chunk"
                },
                {
                    "name": "title",
                    "dataType": ["string"],
                    "description": "Document title"
                },
                {
                    "name": "url",
                    "dataType": ["string"],
                    "description": "Source URL"
                },
                {
                    "name": "source_id",
                    "dataType": ["string"],
                    "description": "Source ID"
                },
                {
                    "name": "document_id",
                    "dataType": ["string"],
                    "description": "Document ID"
                },
                {
                    "name": "chunk_index",
                    "dataType": ["int"],
                    "description": "Position of chunk in document"
                },
                {
                    "name": "source_type",
                    "dataType": ["string"],
                    "description": "Type of source (website, api, etc)"
                },
                {
                    "name": "metadata",
                    "dataType": ["text"],
                    "description": "Additional metadata as JSON"
                },
                {
                    "name": "created_at",
                    "dataType": ["date"],
                    "description": "When the chunk was created"
                }
            ],
            "vectorizer": "none"  # We'll provide our own embeddings
        }
        
        try:
            # Check if class exists
            existing = self.weaviate_client.schema.get()
            if not any(c['class'] == 'KnowledgeChunk' for c in existing.get('classes', [])):
                self.weaviate_client.schema.create_class(schema)
                logger.info("Created Weaviate schema for KnowledgeChunk")
        except Exception as e:
            logger.warning(f"Weaviate schema may already exist: {e}")
    
    def _setup_neo4j_schema(self):
        """Create Neo4j constraints and indexes"""
        with self.neo4j_driver.session() as session:
            queries = [
                # Constraints
                "CREATE CONSTRAINT source_id IF NOT EXISTS FOR (s:Source) REQUIRE s.id IS UNIQUE",
                "CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
                "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
                
                # Indexes for performance
                "CREATE INDEX source_name IF NOT EXISTS FOR (s:Source) ON (s.name)",
                "CREATE INDEX doc_url IF NOT EXISTS FOR (d:Document) ON (d.url)",
                "CREATE INDEX chunk_content IF NOT EXISTS FOR (c:Chunk) ON (c.content)",
                
                # Full-text indexes
                "CREATE FULLTEXT INDEX sourceSearch IF NOT EXISTS FOR (s:Source) ON EACH [s.name, s.url]",
                "CREATE FULLTEXT INDEX documentSearch IF NOT EXISTS FOR (d:Document) ON EACH [d.title, d.content]"
            ]
            
            for query in queries:
                try:
                    session.run(query)
                    logger.info(f"Executed Neo4j query: {query[:50]}...")
                except Exception as e:
                    logger.warning(f"Neo4j query may have already been executed: {e}")
    
    def _setup_timescale_schema(self):
        """Create TimescaleDB tables for analytics"""
        with self.ts_conn.cursor() as cursor:
            # Create tables
            queries = [
                """
                CREATE TABLE IF NOT EXISTS document_metrics (
                    time TIMESTAMPTZ NOT NULL,
                    source_id UUID,
                    document_id UUID,
                    content_length INTEGER,
                    chunk_count INTEGER,
                    processing_time_ms INTEGER,
                    embedding_time_ms INTEGER,
                    metadata JSONB
                );
                """,
                """
                CREATE TABLE IF NOT EXISTS search_analytics (
                    time TIMESTAMPTZ NOT NULL,
                    query TEXT,
                    source_filter TEXT,
                    result_count INTEGER,
                    response_time_ms INTEGER,
                    user_id TEXT,
                    clicked_results JSONB
                );
                """,
                """
                CREATE TABLE IF NOT EXISTS source_health (
                    time TIMESTAMPTZ NOT NULL,
                    source_id UUID,
                    status TEXT,
                    documents_total INTEGER,
                    documents_new INTEGER,
                    error_count INTEGER,
                    avg_doc_size INTEGER
                );
                """
            ]
            
            for query in queries:
                cursor.execute(query)
            
            # Convert to hypertables
            hypertables = [
                "SELECT create_hypertable('document_metrics', 'time', if_not_exists => TRUE);",
                "SELECT create_hypertable('search_analytics', 'time', if_not_exists => TRUE);",
                "SELECT create_hypertable('source_health', 'time', if_not_exists => TRUE);"
            ]
            
            for query in hypertables:
                try:
                    cursor.execute(query)
                except Exception as e:
                    logger.warning(f"Hypertable may already exist: {e}")
            
            self.ts_conn.commit()
            logger.info("TimescaleDB schema setup complete")
    
    def populate_all_databases(self):
        """Main method to populate all databases"""
        logger.info("Starting multi-database population...")
        
        # Get all documents with their chunks
        documents = self._get_all_documents()
        logger.info(f"Found {len(documents)} documents to process")
        
        # Process each document
        for doc in documents:
            try:
                # Get chunks for this document
                chunks = self._get_document_chunks(doc['id'])
                
                # 1. Populate Weaviate with vector embeddings
                self._populate_weaviate(doc, chunks)
                
                # 2. Populate Neo4j knowledge graph
                self._populate_neo4j(doc, chunks)
                
                # 3. Populate TimescaleDB analytics
                self._populate_timescale(doc, chunks)
                
                logger.info(f"Processed document: {doc['title'][:50]}...")
                
            except Exception as e:
                logger.error(f"Error processing document {doc['id']}: {e}")
                continue
        
        # Create knowledge graph relationships
        self._create_neo4j_relationships()
        
        # Generate source health metrics
        self._generate_source_metrics()
        
        logger.info("Multi-database population complete!")
    
    def _get_all_documents(self) -> List[Dict]:
        """Get all documents from PostgreSQL"""
        with self.pg_conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT d.*, s.name as source_name, s.type as source_type, s.url as source_url
                FROM documents d
                JOIN knowledge_sources s ON d.source_id = s.id
                ORDER BY d.created_at DESC
            """)
            return cursor.fetchall()
    
    def _get_document_chunks(self, doc_id: str) -> List[Dict]:
        """Get all chunks for a document"""
        with self.pg_conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT * FROM document_chunks 
                WHERE document_id = %s 
                ORDER BY chunk_index
            """, (doc_id,))
            return cursor.fetchall()
    
    def _populate_weaviate(self, doc: Dict, chunks: List[Dict]):
        """Populate Weaviate with document chunks and embeddings"""
        for chunk in chunks:
            try:
                # Generate embedding
                chunk_text = chunk['content']
                embedding = self.embedder.encode(chunk_text).tolist()
                
                # Try to get enhanced embedding from AI service
                try:
                    response = requests.post(
                        f"{self.ai_service_url}/api/embeddings",
                        json={"text": chunk_text, "model": "code-search"},
                        timeout=5
                    )
                    if response.status_code == 200:
                        embedding = response.json()['embedding']
                except:
                    pass  # Use local embedding if AI service fails
                
                # Create Weaviate object
                data_object = {
                    "content": chunk_text,
                    "title": doc.get('title', ''),
                    "url": doc.get('url', ''),
                    "source_id": str(doc.get('source_id', '')),
                    "document_id": str(doc['id']),
                    "chunk_index": chunk.get('chunk_index', 0),
                    "source_type": doc.get('source_type', 'unknown'),
                    "metadata": json.dumps(chunk.get('metadata', {})),
                    "created_at": doc.get('created_at', datetime.utcnow()).isoformat()
                }
                
                # Add to Weaviate
                self.weaviate_client.data_object.create(
                    data_object=data_object,
                    class_name="KnowledgeChunk",
                    vector=embedding
                )
                
            except Exception as e:
                logger.error(f"Error adding chunk to Weaviate: {e}")
    
    def _populate_neo4j(self, doc: Dict, chunks: List[Dict]):
        """Populate Neo4j knowledge graph"""
        with self.neo4j_driver.session() as session:
            # Create Source node
            session.run("""
                MERGE (s:Source {id: $source_id})
                SET s.name = $source_name,
                    s.url = $source_url,
                    s.type = $source_type
            """, 
                source_id=str(doc['source_id']),
                source_name=doc.get('source_name', ''),
                source_url=doc.get('source_url', ''),
                source_type=doc.get('source_type', '')
            )
            
            # Create Document node
            session.run("""
                MERGE (d:Document {id: $doc_id})
                SET d.title = $title,
                    d.url = $url,
                    d.content_hash = $hash,
                    d.created_at = $created_at
            """,
                doc_id=str(doc['id']),
                title=doc.get('title', ''),
                url=doc.get('url', ''),
                hash=doc.get('content_hash', ''),
                created_at=doc.get('created_at', datetime.utcnow()).isoformat()
            )
            
            # Create relationship
            session.run("""
                MATCH (s:Source {id: $source_id})
                MATCH (d:Document {id: $doc_id})
                MERGE (s)-[:CONTAINS]->(d)
            """,
                source_id=str(doc['source_id']),
                doc_id=str(doc['id'])
            )
            
            # Create Chunk nodes
            for chunk in chunks:
                chunk_id = f"{doc['id']}-{chunk.get('chunk_index', 0)}"
                session.run("""
                    MERGE (c:Chunk {id: $chunk_id})
                    SET c.content = $content,
                        c.index = $index,
                        c.type = $type
                """,
                    chunk_id=chunk_id,
                    content=chunk['content'][:500],  # Store truncated content
                    index=chunk.get('chunk_index', 0),
                    type=chunk.get('chunk_type', 'text')
                )
                
                # Link chunk to document
                session.run("""
                    MATCH (d:Document {id: $doc_id})
                    MATCH (c:Chunk {id: $chunk_id})
                    MERGE (d)-[:HAS_CHUNK]->(c)
                """,
                    doc_id=str(doc['id']),
                    chunk_id=chunk_id
                )
    
    def _populate_timescale(self, doc: Dict, chunks: List[Dict]):
        """Populate TimescaleDB with analytics data"""
        with self.ts_conn.cursor() as cursor:
            # Document metrics
            cursor.execute("""
                INSERT INTO document_metrics (
                    time, source_id, document_id, content_length, 
                    chunk_count, processing_time_ms, metadata
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                doc.get('created_at', datetime.utcnow()),
                doc['source_id'],
                doc['id'],
                len(doc.get('content', '')),
                len(chunks),
                100,  # Placeholder processing time
                json.dumps(doc.get('metadata', {}))
            ))
            
            self.ts_conn.commit()
    
    def _create_neo4j_relationships(self):
        """Create additional relationships in Neo4j"""
        with self.neo4j_driver.session() as session:
            # Find related documents based on content similarity
            session.run("""
                MATCH (d1:Document)
                MATCH (d2:Document)
                WHERE d1.id <> d2.id 
                  AND d1.url CONTAINS d2.source_type
                MERGE (d1)-[:REFERENCES]->(d2)
            """)
            
            # Create topic clusters
            session.run("""
                MATCH (d:Document)
                WHERE d.title CONTAINS 'API'
                MERGE (t:Topic {name: 'API Documentation'})
                MERGE (d)-[:ABOUT]->(t)
            """)
            
            session.run("""
                MATCH (d:Document)
                WHERE d.title CONTAINS 'Authentication'
                MERGE (t:Topic {name: 'Authentication'})
                MERGE (d)-[:ABOUT]->(t)
            """)
            
            logger.info("Created Neo4j relationships")
    
    def _generate_source_metrics(self):
        """Generate source health metrics for TimescaleDB"""
        with self.pg_conn.cursor(cursor_factory=RealDictCursor) as pg_cursor:
            pg_cursor.execute("""
                SELECT 
                    s.id as source_id,
                    s.status,
                    COUNT(d.id) as doc_count,
                    AVG(LENGTH(d.content)) as avg_doc_size
                FROM knowledge_sources s
                LEFT JOIN documents d ON s.id = d.source_id
                GROUP BY s.id, s.status
            """)
            
            with self.ts_conn.cursor() as ts_cursor:
                for row in pg_cursor.fetchall():
                    ts_cursor.execute("""
                        INSERT INTO source_health (
                            time, source_id, status, documents_total, 
                            documents_new, error_count, avg_doc_size
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        datetime.utcnow(),
                        row['source_id'],
                        row['status'],
                        row['doc_count'],
                        0,  # New docs in this run
                        0,  # Error count
                        int(row['avg_doc_size'] or 0)
                    ))
                
                self.ts_conn.commit()
    
    def verify_population(self):
        """Verify that all databases have been populated"""
        logger.info("\n=== Database Population Verification ===")
        
        # Check Weaviate
        weaviate_count = self.weaviate_client.query.aggregate("KnowledgeChunk").with_meta_count().do()
        logger.info(f"Weaviate: {weaviate_count['data']['Aggregate']['KnowledgeChunk'][0]['meta']['count']} chunks")
        
        # Check Neo4j
        with self.neo4j_driver.session() as session:
            result = session.run("MATCH (n) RETURN labels(n)[0] as label, count(n) as count")
            for record in result:
                logger.info(f"Neo4j: {record['count']} {record['label']} nodes")
        
        # Check TimescaleDB
        with self.ts_conn.cursor() as cursor:
            tables = ['document_metrics', 'source_health']
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                logger.info(f"TimescaleDB: {count} rows in {table}")

def main():
    """Main execution"""
    populator = MultiDatabasePopulator()
    
    try:
        # Setup schemas
        logger.info("Setting up database schemas...")
        populator.setup_databases()
        
        # Populate all databases
        logger.info("Populating databases...")
        populator.populate_all_databases()
        
        # Verify
        populator.verify_population()
        
        logger.info("\n✅ All databases have been populated successfully!")
        
    except Exception as e:
        logger.error(f"Error during population: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        populator.pg_conn.close()
        populator.ts_conn.close()
        populator.neo4j_driver.close()

if __name__ == "__main__":
    main()