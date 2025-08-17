"""
MCP API Integration Layer for Advanced AI Analysis
Created by Annelies Claes - Expert in API Design & MCP Integration

This module provides seamless integration between the advanced AI analysis services
and the existing KnowledgeHub MCP server architecture.
"""

import asyncio
import logging
import json
import httpx
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from contextlib import asynccontextmanager

from .quantized_ai_service import (
    QuantizedAIService, 
    ContentAnalysisRequest, 
    SemanticSimilarityRequest,
    UserBehaviorAnalysisRequest
)

logger = logging.getLogger(__name__)

class MCPAIIntegration:
    """
    Integration layer for connecting Advanced AI Analysis with KnowledgeHub MCP servers.
    
    This class orchestrates communication between:
    - Context7 MCP (documentation patterns)
    - Sequential MCP (complex reasoning)
    - Database MCP (data persistence)
    - Playwright MCP (testing validation)
    """
    
    def __init__(
        self,
        knowledgehub_base: str = "http://192.168.1.25:3000",
        ai_service_base: str = "http://192.168.1.25:8003",
        synology_mcp_base: str = "http://192.168.1.24"
    ):
        self.knowledgehub_base = knowledgehub_base
        self.ai_service_base = ai_service_base
        self.synology_mcp_base = synology_mcp_base
        
        # MCP service endpoints
        self.mcp_endpoints = {
            'context7': f"{synology_mcp_base}:3007",  # Documentation patterns
            'sequential': f"{synology_mcp_base}:3008",  # Complex reasoning
            'database': f"{synology_mcp_base}:3011",   # Database operations
            'playwright': f"{synology_mcp_base}:3013",  # Testing validation
            'knowledgehub_mcp': f"{synology_mcp_base}:3008"  # KnowledgeHub MCP
        }
        
        self.ai_service: Optional[QuantizedAIService] = None
        self.http_client: Optional[httpx.AsyncClient] = None

    async def initialize(self):
        """Initialize the integration layer and all services."""
        try:
            # Initialize AI service
            self.ai_service = QuantizedAIService(
                knowledgehub_api_base=self.knowledgehub_base,
                ai_service_base=self.ai_service_base
            )
            await self.ai_service.initialize()
            
            # Initialize HTTP client for MCP communication
            self.http_client = httpx.AsyncClient(timeout=30.0)
            
            # Validate MCP endpoints
            await self._validate_mcp_endpoints()
            
            logger.info("MCP AI Integration initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP AI Integration: {e}")
            raise

    async def _validate_mcp_endpoints(self):
        """Validate connectivity to MCP endpoints."""
        for service_name, endpoint in self.mcp_endpoints.items():
            try:
                response = await self.http_client.get(f"{endpoint}/health", timeout=5.0)
                if response.status_code == 200:
                    logger.debug(f"MCP service {service_name} is healthy")
                else:
                    logger.warning(f"MCP service {service_name} returned status {response.status_code}")
            except Exception as e:
                logger.warning(f"MCP service {service_name} is not accessible: {e}")

    async def enhanced_content_analysis(
        self,
        content: str,
        content_type: str = "text",
        use_mcp_context: bool = True,
        analysis_depth: str = "standard"
    ) -> Dict[str, Any]:
        """
        Enhanced content analysis using AI service + MCP context.
        
        Integrates:
        1. Quantized AI pattern recognition
        2. Context7 MCP for documentation patterns
        3. Sequential MCP for complex reasoning
        4. Database MCP for historical context
        """
        analysis_start = datetime.utcnow()
        
        try:
            # Step 1: Get MCP context if requested
            mcp_context = {}
            if use_mcp_context:
                mcp_context = await self._gather_mcp_context(content, content_type)
            
            # Step 2: Perform AI analysis with MCP context
            ai_request = ContentAnalysisRequest(
                content=content,
                content_type=content_type,
                analysis_depth=analysis_depth,
                context=mcp_context
            )
            
            ai_response = await self.ai_service.analyze_content(ai_request)
            
            # Step 3: Enhance results with MCP-specific insights
            enhanced_results = await self._enhance_with_mcp_insights(ai_response, mcp_context)
            
            # Step 4: Store results in KnowledgeHub for learning
            await self._store_analysis_results(enhanced_results)
            
            processing_time = (datetime.utcnow() - analysis_start).total_seconds()
            
            return {
                "analysis_id": enhanced_results.analysis_id,
                "content_analysis": {
                    "patterns": [p.dict() for p in enhanced_results.patterns],
                    "summary": enhanced_results.summary,
                    "recommendations": enhanced_results.recommendations,
                    "confidence_score": enhanced_results.confidence_score
                },
                "mcp_insights": mcp_context,
                "performance": {
                    "ai_processing_time": enhanced_results.processing_time,
                    "total_processing_time": processing_time,
                    "mcp_integration_overhead": processing_time - enhanced_results.processing_time
                },
                "metadata": {
                    "analysis_timestamp": analysis_start.isoformat(),
                    "ai_model_info": enhanced_results.performance_metrics,
                    "mcp_services_used": list(mcp_context.keys()) if mcp_context else []
                }
            }
            
        except Exception as e:
            logger.error(f"Enhanced content analysis failed: {e}")
            raise

    async def _gather_mcp_context(self, content: str, content_type: str) -> Dict[str, Any]:
        """Gather context from relevant MCP services."""
        context = {}
        
        try:
            # Context7: Get documentation patterns
            if content_type in ['code', 'text']:
                doc_context = await self._get_context7_insights(content)
                if doc_context:
                    context['documentation_patterns'] = doc_context
            
            # Sequential: Get complex reasoning insights
            reasoning_context = await self._get_sequential_insights(content)
            if reasoning_context:
                context['reasoning_analysis'] = reasoning_context
            
            # Database: Get historical analysis context
            historical_context = await self._get_database_context(content)
            if historical_context:
                context['historical_analysis'] = historical_context
                
        except Exception as e:
            logger.warning(f"Failed to gather MCP context: {e}")
        
        return context

    async def _get_context7_insights(self, content: str) -> Optional[Dict[str, Any]]:
        """Get documentation and pattern insights from Context7 MCP."""
        try:
            # Context7 MCP call for documentation patterns
            payload = {
                "method": "get-library-docs",
                "params": {
                    "content": content[:1000],  # Limit content size
                    "focus": "patterns"
                }
            }
            
            response = await self.http_client.post(
                f"{self.mcp_endpoints['context7']}/api/context",
                json=payload,
                timeout=10.0
            )
            
            if response.status_code == 200:
                return response.json()
                
        except Exception as e:
            logger.debug(f"Context7 MCP call failed: {e}")
        
        return None

    async def _get_sequential_insights(self, content: str) -> Optional[Dict[str, Any]]:
        """Get complex reasoning insights from Sequential MCP."""
        try:
            # Sequential MCP call for complex analysis
            payload = {
                "method": "analyze-complexity",
                "params": {
                    "content": content,
                    "analysis_type": "pattern_complexity"
                }
            }
            
            response = await self.http_client.post(
                f"{self.mcp_endpoints['sequential']}/api/sequential",
                json=payload,
                timeout=15.0
            )
            
            if response.status_code == 200:
                return response.json()
                
        except Exception as e:
            logger.debug(f"Sequential MCP call failed: {e}")
        
        return None

    async def _get_database_context(self, content: str) -> Optional[Dict[str, Any]]:
        """Get historical analysis context from Database MCP."""
        try:
            # Database MCP call for historical data
            content_hash = str(hash(content))
            payload = {
                "method": "query-similar-analyses",
                "params": {
                    "content_hash": content_hash,
                    "similarity_threshold": 0.8,
                    "limit": 5
                }
            }
            
            response = await self.http_client.post(
                f"{self.mcp_endpoints['database']}/api/database",
                json=payload,
                timeout=10.0
            )
            
            if response.status_code == 200:
                return response.json()
                
        except Exception as e:
            logger.debug(f"Database MCP call failed: {e}")
        
        return None

    async def _enhance_with_mcp_insights(self, ai_response, mcp_context: Dict[str, Any]):
        """Enhance AI analysis results with MCP insights."""
        # Add MCP-specific recommendations
        enhanced_recommendations = ai_response.recommendations.copy()
        
        # Add Context7 recommendations
        if 'documentation_patterns' in mcp_context:
            doc_patterns = mcp_context['documentation_patterns']
            if 'missing_documentation' in doc_patterns:
                enhanced_recommendations.append("Add missing documentation based on detected patterns")
        
        # Add Sequential reasoning recommendations
        if 'reasoning_analysis' in mcp_context:
            reasoning = mcp_context['reasoning_analysis']
            if reasoning.get('complexity_score', 0) > 0.8:
                enhanced_recommendations.append("Consider refactoring complex logic for better maintainability")
        
        # Add historical context recommendations
        if 'historical_analysis' in mcp_context:
            historical = mcp_context['historical_analysis']
            if historical.get('similar_issues_found', False):
                enhanced_recommendations.append("Similar issues have been detected before - check historical solutions")
        
        # Update the response
        ai_response.recommendations = enhanced_recommendations
        return ai_response

    async def _store_analysis_results(self, analysis_results) -> None:
        """Store analysis results in KnowledgeHub for learning."""
        try:
            storage_payload = {
                "analysis_id": analysis_results.analysis_id,
                "patterns": [p.dict() for p in analysis_results.patterns],
                "confidence_score": analysis_results.confidence_score,
                "recommendations": analysis_results.recommendations,
                "timestamp": datetime.utcnow().isoformat(),
                "source": "mcp_ai_integration"
            }
            
            await self.http_client.post(
                f"{self.knowledgehub_base}/api/claude-auto/store-analysis",
                json=storage_payload,
                timeout=10.0
            )
            
        except Exception as e:
            logger.warning(f"Failed to store analysis results: {e}")

    async def advanced_document_similarity(
        self,
        query: str,
        document_ids: List[str],
        similarity_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Advanced document similarity analysis integrating multiple MCP services.
        
        Goes beyond basic RAG by incorporating:
        1. Semantic similarity (AI service)
        2. Structural similarity (Context7 patterns)
        3. Historical similarity (Database MCP)
        4. Content quality metrics
        """
        try:
            # Step 1: Get documents from KnowledgeHub
            documents = await self._fetch_documents(document_ids)
            
            # Step 2: Perform AI-based semantic similarity
            similarity_request = SemanticSimilarityRequest(
                query_content=query,
                target_contents=[doc['content'] for doc in documents],
                similarity_threshold=similarity_threshold,
                use_quantized_model=True
            )
            
            ai_similarity = await self.ai_service.semantic_similarity_analysis(similarity_request)
            
            # Step 3: Enhance with MCP-based analysis
            enhanced_matches = []
            for match in ai_similarity.matches:
                doc_index = int(match.content_id.split('_')[1])
                document = documents[doc_index]
                
                # Get Context7 structural analysis
                structural_score = await self._get_structural_similarity(query, document['content'])
                
                # Get historical context
                historical_score = await self._get_historical_similarity(query, document['id'])
                
                # Calculate enhanced similarity score
                enhanced_score = (
                    0.5 * match.similarity_score +
                    0.3 * structural_score +
                    0.2 * historical_score
                )
                
                enhanced_matches.append({
                    "document_id": document['id'],
                    "title": document.get('title', 'Untitled'),
                    "semantic_similarity": match.similarity_score,
                    "structural_similarity": structural_score,
                    "historical_similarity": historical_score,
                    "enhanced_similarity": enhanced_score,
                    "match_type": match.match_type,
                    "confidence": match.confidence,
                    "content_preview": match.content_preview,
                    "metadata": document.get('metadata', {})
                })
            
            # Sort by enhanced similarity
            enhanced_matches.sort(key=lambda x: x['enhanced_similarity'], reverse=True)
            
            return {
                "query": query,
                "matches": enhanced_matches,
                "analysis_info": {
                    "total_documents_analyzed": len(documents),
                    "matches_found": len(enhanced_matches),
                    "similarity_threshold": similarity_threshold,
                    "processing_time": ai_similarity.processing_time,
                    "model_info": ai_similarity.model_info
                },
                "mcp_integration": {
                    "structural_analysis_enabled": True,
                    "historical_context_enabled": True,
                    "enhancement_algorithm": "weighted_multi_factor"
                }
            }
            
        except Exception as e:
            logger.error(f"Advanced document similarity failed: {e}")
            raise

    async def _fetch_documents(self, document_ids: List[str]) -> List[Dict[str, Any]]:
        """Fetch documents from KnowledgeHub."""
        documents = []
        
        for doc_id in document_ids:
            try:
                response = await self.http_client.get(
                    f"{self.knowledgehub_base}/api/documents/{doc_id}",
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    documents.append(response.json())
                    
            except Exception as e:
                logger.warning(f"Failed to fetch document {doc_id}: {e}")
        
        return documents

    async def _get_structural_similarity(self, query: str, content: str) -> float:
        """Get structural similarity using Context7 patterns."""
        try:
            payload = {
                "method": "compare-structures",
                "params": {
                    "query": query,
                    "content": content
                }
            }
            
            response = await self.http_client.post(
                f"{self.mcp_endpoints['context7']}/api/structural-analysis",
                json=payload,
                timeout=10.0
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('similarity_score', 0.0)
                
        except Exception as e:
            logger.debug(f"Structural similarity calculation failed: {e}")
        
        return 0.0

    async def _get_historical_similarity(self, query: str, document_id: str) -> float:
        """Get historical similarity using Database MCP."""
        try:
            payload = {
                "method": "query-interaction-history",
                "params": {
                    "query": query,
                    "document_id": document_id
                }
            }
            
            response = await self.http_client.post(
                f"{self.mcp_endpoints['database']}/api/history-analysis",
                json=payload,
                timeout=10.0
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('historical_relevance_score', 0.0)
                
        except Exception as e:
            logger.debug(f"Historical similarity calculation failed: {e}")
        
        return 0.0

    async def real_time_intelligence_pipeline(
        self,
        user_id: str,
        session_data: List[Dict[str, Any]],
        content_stream: str
    ) -> Dict[str, Any]:
        """
        Real-time AI intelligence pipeline combining multiple analysis types.
        
        Provides:
        1. Real-time pattern recognition
        2. User behavior analysis
        3. Content quality assessment
        4. Proactive recommendations
        """
        pipeline_start = datetime.utcnow()
        
        try:
            # Parallel analysis tasks
            tasks = [
                # Content analysis
                self.enhanced_content_analysis(
                    content=content_stream,
                    content_type="text",
                    use_mcp_context=True,
                    analysis_depth="fast"  # Use fast mode for real-time
                ),
                
                # User behavior analysis
                self.ai_service.analyze_user_behavior(
                    UserBehaviorAnalysisRequest(
                        user_id=user_id,
                        session_data=session_data,
                        time_window=3600,
                        analysis_type="real_time_patterns"
                    )
                )
            ]
            
            # Execute tasks in parallel
            content_analysis, behavior_analysis = await asyncio.gather(*tasks)
            
            # Generate unified insights
            unified_insights = await self._generate_unified_insights(
                content_analysis, behavior_analysis
            )
            
            # Generate proactive recommendations
            proactive_recommendations = await self._generate_proactive_recommendations(
                content_analysis, behavior_analysis, unified_insights
            )
            
            processing_time = (datetime.utcnow() - pipeline_start).total_seconds()
            
            return {
                "user_id": user_id,
                "real_time_analysis": {
                    "content_insights": content_analysis,
                    "behavior_patterns": behavior_analysis.dict(),
                    "unified_insights": unified_insights,
                    "proactive_recommendations": proactive_recommendations
                },
                "performance": {
                    "total_processing_time": processing_time,
                    "real_time_capable": processing_time < 2.0,  # Under 2 seconds for real-time
                    "parallel_execution": True
                },
                "metadata": {
                    "pipeline_timestamp": pipeline_start.isoformat(),
                    "ai_model_efficiency": content_analysis.get("performance", {}),
                    "mcp_integration_active": True
                }
            }
            
        except Exception as e:
            logger.error(f"Real-time intelligence pipeline failed: {e}")
            raise

    async def _generate_unified_insights(
        self,
        content_analysis: Dict[str, Any],
        behavior_analysis
    ) -> Dict[str, Any]:
        """Generate unified insights from multiple analysis results."""
        insights = {
            "overall_risk_score": 0.0,
            "user_engagement_level": "medium",
            "content_quality_score": 0.0,
            "anomaly_indicators": [],
            "pattern_correlations": [],
            "learning_opportunities": []
        }
        
        # Calculate overall risk score
        content_patterns = content_analysis.get("content_analysis", {}).get("patterns", [])
        critical_patterns = [p for p in content_patterns if p.get("severity") == "critical"]
        high_patterns = [p for p in content_patterns if p.get("severity") == "high"]
        
        risk_score = len(critical_patterns) * 0.4 + len(high_patterns) * 0.2
        insights["overall_risk_score"] = min(risk_score, 1.0)
        
        # Determine user engagement level
        behavior_patterns = behavior_analysis.patterns
        if len(behavior_patterns) > 3:
            insights["user_engagement_level"] = "high"
        elif len(behavior_patterns) < 1:
            insights["user_engagement_level"] = "low"
        
        # Calculate content quality score
        confidence_score = content_analysis.get("content_analysis", {}).get("confidence_score", 0.0)
        pattern_count = len(content_patterns)
        quality_score = confidence_score * 0.7 + min(pattern_count / 10, 1.0) * 0.3
        insights["content_quality_score"] = quality_score
        
        # Identify anomalies
        if behavior_analysis.anomalies:
            insights["anomaly_indicators"] = [a["type"] for a in behavior_analysis.anomalies]
        
        # Find pattern correlations
        if len(content_patterns) > 1 and len(behavior_patterns) > 1:
            insights["pattern_correlations"] = [
                "content_behavior_correlation_detected"
            ]
        
        return insights

    async def _generate_proactive_recommendations(
        self,
        content_analysis: Dict[str, Any],
        behavior_analysis,
        unified_insights: Dict[str, Any]
    ) -> List[str]:
        """Generate proactive recommendations based on all analyses."""
        recommendations = []
        
        # Risk-based recommendations
        if unified_insights["overall_risk_score"] > 0.7:
            recommendations.append("HIGH PRIORITY: Address critical security or quality issues immediately")
        
        # Engagement-based recommendations
        if unified_insights["user_engagement_level"] == "low":
            recommendations.append("Consider personalized content recommendations to improve engagement")
        elif unified_insights["user_engagement_level"] == "high":
            recommendations.append("User is highly engaged - consider advanced features or premium content")
        
        # Quality-based recommendations
        if unified_insights["content_quality_score"] < 0.5:
            recommendations.append("Content quality is below average - review and improve content standards")
        
        # Anomaly-based recommendations
        if unified_insights["anomaly_indicators"]:
            recommendations.append("Anomalous behavior detected - monitor for potential security issues")
        
        # Pattern correlation recommendations
        if unified_insights["pattern_correlations"]:
            recommendations.append("Strong correlations detected between content and behavior patterns")
        
        # Add content-specific recommendations
        content_recommendations = content_analysis.get("content_analysis", {}).get("recommendations", [])
        recommendations.extend(content_recommendations[:3])  # Top 3 content recommendations
        
        # Add behavior-specific recommendations
        behavior_recommendations = behavior_analysis.recommendations
        recommendations.extend(behavior_recommendations[:2])  # Top 2 behavior recommendations
        
        return recommendations[:10]  # Limit to top 10 recommendations

    async def validate_with_playwright(
        self,
        analysis_results: Dict[str, Any],
        validation_scenarios: List[str]
    ) -> Dict[str, Any]:
        """
        Validate AI analysis results using Playwright MCP for end-to-end testing.
        """
        try:
            validation_results = {
                "validation_id": f"validation_{int(datetime.utcnow().timestamp())}",
                "scenario_results": [],
                "overall_validation_status": "pending",
                "validation_coverage": 0.0
            }
            
            for scenario in validation_scenarios:
                try:
                    # Call Playwright MCP for validation
                    payload = {
                        "method": "validate-analysis",
                        "params": {
                            "analysis_data": analysis_results,
                            "validation_scenario": scenario
                        }
                    }
                    
                    response = await self.http_client.post(
                        f"{self.mcp_endpoints['playwright']}/api/validate",
                        json=payload,
                        timeout=30.0
                    )
                    
                    if response.status_code == 200:
                        scenario_result = response.json()
                        scenario_result["scenario"] = scenario
                        validation_results["scenario_results"].append(scenario_result)
                    else:
                        validation_results["scenario_results"].append({
                            "scenario": scenario,
                            "status": "failed",
                            "error": f"HTTP {response.status_code}"
                        })
                        
                except Exception as e:
                    validation_results["scenario_results"].append({
                        "scenario": scenario,
                        "status": "error",
                        "error": str(e)
                    })
            
            # Calculate overall validation status
            successful_validations = sum(
                1 for r in validation_results["scenario_results"] 
                if r.get("status") == "passed"
            )
            
            total_validations = len(validation_results["scenario_results"])
            validation_results["validation_coverage"] = (
                successful_validations / total_validations if total_validations > 0 else 0.0
            )
            
            if validation_results["validation_coverage"] >= 0.8:
                validation_results["overall_validation_status"] = "passed"
            elif validation_results["validation_coverage"] >= 0.6:
                validation_results["overall_validation_status"] = "partial"
            else:
                validation_results["overall_validation_status"] = "failed"
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Playwright validation failed: {e}")
            return {
                "validation_id": f"validation_error_{int(datetime.utcnow().timestamp())}",
                "overall_validation_status": "error",
                "error": str(e)
            }

    async def get_integration_health(self) -> Dict[str, Any]:
        """Get comprehensive health status of the MCP integration."""
        health_status = {
            "service": "MCP AI Integration",
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {},
            "mcp_services": {},
            "performance_metrics": {}
        }
        
        try:
            # Check AI service health
            if self.ai_service:
                ai_health = await self.ai_service.get_service_health()
                health_status["components"]["ai_service"] = ai_health
            else:
                health_status["components"]["ai_service"] = {"status": "not_initialized"}
                health_status["status"] = "degraded"
            
            # Check MCP service connectivity
            for service_name, endpoint in self.mcp_endpoints.items():
                try:
                    response = await self.http_client.get(f"{endpoint}/health", timeout=5.0)
                    health_status["mcp_services"][service_name] = {
                        "status": "operational" if response.status_code == 200 else "degraded",
                        "endpoint": endpoint,
                        "response_code": response.status_code
                    }
                except Exception as e:
                    health_status["mcp_services"][service_name] = {
                        "status": "error",
                        "endpoint": endpoint,
                        "error": str(e)
                    }
            
            # Calculate overall status
            mcp_healthy = sum(
                1 for s in health_status["mcp_services"].values() 
                if s["status"] == "operational"
            )
            mcp_total = len(health_status["mcp_services"])
            
            if mcp_healthy < mcp_total * 0.5:  # Less than 50% MCP services healthy
                health_status["status"] = "degraded"
            
            health_status["performance_metrics"] = {
                "mcp_services_operational": mcp_healthy,
                "mcp_services_total": mcp_total,
                "mcp_service_availability": mcp_healthy / mcp_total if mcp_total > 0 else 0.0
            }
            
        except Exception as e:
            health_status["status"] = "error"
            health_status["error"] = str(e)
        
        return health_status

    async def close(self):
        """Clean up resources."""
        if self.http_client:
            await self.http_client.aclose()

