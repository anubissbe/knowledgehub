#!/usr/bin/env python3
"""
KnowledgeHub AI Service
Provides AI-powered analysis, threat detection, and content insights
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx
import asyncpg
import json
from sentence_transformers import SentenceTransformer
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pydantic Models
class ThreatAnalysisRequest(BaseModel):
    content: str = Field(..., description="Content to analyze for threats")
    context: Optional[str] = Field(None, description="Additional context")
    analysis_type: str = Field("general", description="Type of analysis (general, security, compliance)")

class ThreatAnalysisResponse(BaseModel):
    threats: List[Dict[str, Any]]
    risk_score: float = Field(..., ge=0.0, le=10.0)
    recommendations: List[str]
    confidence: float = Field(..., ge=0.0, le=1.0)
    analysis_id: str

class ContentSimilarityRequest(BaseModel):
    query: str = Field(..., description="Query text to find similar content")
    limit: int = Field(10, ge=1, le=100)
    threshold: float = Field(0.7, ge=0.0, le=1.0)

class ContentSimilarityResponse(BaseModel):
    similar_content: List[Dict[str, Any]]
    query_embedding: Optional[List[float]] = None

class RiskScoringRequest(BaseModel):
    components: List[str] = Field(..., description="System components to analyze")
    data_flows: List[Dict[str, str]] = Field(..., description="Data flow descriptions")
    environment: str = Field("production", description="Environment type")

class RiskScoringResponse(BaseModel):
    overall_risk: float = Field(..., ge=0.0, le=10.0)
    component_risks: Dict[str, float]
    risk_factors: List[Dict[str, Any]]
    mitigation_suggestions: List[str]

# Global variables
embedding_model = None
db_pool = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global embedding_model, db_pool
    
    logger.info("Initializing AI Service...")
    
    # Load embedding model
    try:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Embedding model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        raise
    
    # Database connection
    try:
        db_pool = await asyncpg.create_pool(
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', 5432)),  # Within Docker network, postgres is on 5432
            database=os.getenv('DB_NAME', 'knowledgehub'),
            user=os.getenv('DB_USER', 'knowledgehub'),
            password=os.getenv('DB_PASSWORD', 'knowledgehub123'),
            min_size=5,
            max_size=20
        )
        logger.info("Database connection pool created")
    except Exception as e:
        logger.error(f"Failed to create database pool: {e}")
        logger.warning("Continuing without database connection for now...")
        db_pool = None  # Allow service to start without database
    
    # Set app state
    app.state.embedding_model = embedding_model
    app.state.db_pool = db_pool
    
    yield
    
    # Cleanup
    if db_pool:
        await db_pool.close()
        logger.info("Database pool closed")

# FastAPI app
app = FastAPI(
    title="KnowledgeHub AI Service",
    description="AI-powered analysis and threat detection for KnowledgeHub",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "ai_service": "operational",
            "embedding_model": "loaded" if embedding_model else "not_loaded",
            "database": "connected" if db_pool else "disconnected"
        }
    }
    
    # Check database connection
    if db_pool:
        try:
            async with db_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            health_status["services"]["database"] = "operational"
        except Exception as e:
            health_status["services"]["database"] = f"error: {str(e)}"
            health_status["status"] = "degraded"
    
    return health_status

# AI Service Endpoints

@app.post("/api/ai/analyze-threats", response_model=ThreatAnalysisResponse)
async def analyze_threats(request: ThreatAnalysisRequest):
    """Analyze content for potential security threats"""
    try:
        analysis_id = f"threat-{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Simple threat detection patterns (can be enhanced with ML models)
        threat_patterns = {
            "sql_injection": ["SELECT", "INSERT", "DELETE", "DROP", "UNION", "OR 1=1"],
            "xss": ["<script>", "javascript:", "onerror=", "onload="],
            "path_traversal": ["../", "..\\", "%2e%2e"],
            "command_injection": ["&&", "||", ";", "|", "`"],
            "sensitive_data": ["password", "secret", "token", "key", "credential"]
        }
        
        threats = []
        risk_score = 0.0
        content_lower = request.content.lower()
        
        for threat_type, patterns in threat_patterns.items():
            for pattern in patterns:
                if pattern.lower() in content_lower:
                    severity = "high" if threat_type in ["sql_injection", "command_injection"] else "medium"
                    threat_risk = 8.0 if severity == "high" else 5.0
                    
                    threats.append({
                        "type": threat_type,
                        "pattern": pattern,
                        "severity": severity,
                        "risk_score": threat_risk,
                        "description": f"Potential {threat_type.replace('_', ' ')} detected"
                    })
                    
                    risk_score = max(risk_score, threat_risk)
        
        # Generate recommendations
        recommendations = []
        if threats:
            recommendations.append("Review and sanitize the identified suspicious patterns")
            recommendations.append("Implement input validation and output encoding")
            recommendations.append("Use parameterized queries for database operations")
        else:
            recommendations.append("No immediate threats detected, but continue monitoring")
        
        confidence = 0.8 if threats else 0.9
        
        return ThreatAnalysisResponse(
            threats=threats,
            risk_score=risk_score,
            recommendations=recommendations,
            confidence=confidence,
            analysis_id=analysis_id
        )
        
    except Exception as e:
        logger.error(f"Error in threat analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ai/content-similarity", response_model=ContentSimilarityResponse)
async def find_similar_content(request: ContentSimilarityRequest):
    """Find similar content using semantic search"""
    try:
        if not embedding_model:
            raise HTTPException(status_code=500, detail="Embedding model not loaded")
        
        # Generate query embedding
        query_embedding = embedding_model.encode(request.query).tolist()
        
        # Search for similar content in database
        similar_content = []
        
        if db_pool:
            async with db_pool.acquire() as conn:
                # Get existing chunks with embeddings (if available)
                rows = await conn.fetch("""
                    SELECT dc.id, dc.content, dc.metadata, d.url, d.title
                    FROM document_chunks dc
                    JOIN documents d ON dc.document_id = d.id
                    WHERE LENGTH(dc.content) > 50
                    ORDER BY dc.created_at DESC
                    LIMIT 100
                """)
                
                for row in rows:
                    # Calculate similarity
                    content_embedding = embedding_model.encode(row['content'])
                    similarity = np.dot(query_embedding, content_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(content_embedding)
                    )
                    
                    if similarity >= request.threshold:
                        similar_content.append({
                            "chunk_id": str(row['id']),
                            "content": row['content'][:500] + "..." if len(row['content']) > 500 else row['content'],
                            "similarity": float(similarity),
                            "url": row['url'],
                            "title": row['title'],
                            "metadata": row['metadata'] or {}
                        })
                
                # Sort by similarity and limit results
                similar_content.sort(key=lambda x: x['similarity'], reverse=True)
                similar_content = similar_content[:request.limit]
        
        return ContentSimilarityResponse(
            similar_content=similar_content,
            query_embedding=query_embedding if len(similar_content) > 0 else None
        )
        
    except Exception as e:
        logger.error(f"Error in similarity search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ai/risk-scoring", response_model=RiskScoringResponse)
async def calculate_risk_score(request: RiskScoringRequest):
    """Calculate risk scores for system components and data flows"""
    try:
        # Risk factors and weights
        risk_factors = {
            "external_facing": 3.0,
            "database_access": 2.5,
            "user_input": 2.0,
            "authentication": 1.5,
            "encryption": -1.0,  # Reduces risk
            "monitoring": -0.5,  # Reduces risk
        }
        
        component_risks = {}
        overall_risk = 0.0
        identified_factors = []
        
        # Analyze each component
        for component in request.components:
            component_risk = 5.0  # Base risk
            component_lower = component.lower()
            
            # Check for risk indicators
            if any(term in component_lower for term in ["web", "api", "frontend", "public"]):
                component_risk += risk_factors["external_facing"]
                identified_factors.append({"factor": "external_facing", "component": component, "impact": 3.0})
            
            if any(term in component_lower for term in ["database", "db", "sql", "storage"]):
                component_risk += risk_factors["database_access"]
                identified_factors.append({"factor": "database_access", "component": component, "impact": 2.5})
            
            if any(term in component_lower for term in ["form", "input", "user", "upload"]):
                component_risk += risk_factors["user_input"]
                identified_factors.append({"factor": "user_input", "component": component, "impact": 2.0})
            
            # Environment adjustments
            if request.environment == "production":
                component_risk += 1.0
            elif request.environment == "development":
                component_risk -= 1.0
            
            # Ensure risk is within bounds
            component_risk = max(0.0, min(10.0, component_risk))
            component_risks[component] = component_risk
            overall_risk += component_risk
        
        # Calculate average risk
        if component_risks:
            overall_risk = overall_risk / len(component_risks)
        
        # Analyze data flows for additional risks
        for flow in request.data_flows:
            flow_desc = flow.get('description', '').lower()
            if any(term in flow_desc for term in ["internet", "external", "public"]):
                overall_risk += 0.5
                identified_factors.append({"factor": "external_data_flow", "description": flow_desc, "impact": 0.5})
        
        # Ensure overall risk is within bounds
        overall_risk = max(0.0, min(10.0, overall_risk))
        
        # Generate mitigation suggestions
        mitigation_suggestions = [
            "Implement comprehensive input validation",
            "Use encryption for data transmission and storage",
            "Deploy monitoring and alerting systems",
            "Regular security assessments and penetration testing",
            "Implement the principle of least privilege"
        ]
        
        # Add specific suggestions based on identified risks
        if any(f["factor"] == "external_facing" for f in identified_factors):
            mitigation_suggestions.append("Implement Web Application Firewall (WAF)")
        
        if any(f["factor"] == "database_access" for f in identified_factors):
            mitigation_suggestions.append("Use parameterized queries and database access controls")
        
        return RiskScoringResponse(
            overall_risk=overall_risk,
            component_risks=component_risks,
            risk_factors=identified_factors,
            mitigation_suggestions=mitigation_suggestions
        )
        
    except Exception as e:
        logger.error(f"Error in risk scoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Additional utility endpoints

@app.get("/api/ai/models/status")
async def get_model_status():
    """Get status of loaded AI models"""
    return {
        "embedding_model": {
            "loaded": embedding_model is not None,
            "model_name": "all-MiniLM-L6-v2" if embedding_model else None,
            "dimensions": 384 if embedding_model else None
        }
    }

@app.post("/api/ai/embed")
async def generate_embeddings(content: Dict[str, str]):
    """Generate embeddings for given content"""
    try:
        if not embedding_model:
            raise HTTPException(status_code=500, detail="Embedding model not loaded")
        
        text = content.get("text", "")
        if not text:
            raise HTTPException(status_code=400, detail="Text content required")
        
        embedding = embedding_model.encode(text).tolist()
        
        return {
            "embedding": embedding,
            "dimensions": len(embedding),
            "model": "all-MiniLM-L6-v2"
        }
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )