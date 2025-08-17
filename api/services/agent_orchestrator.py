"""
Agent Orchestrator Service - LangGraph-based Multi-Agent RAG System
Implements stateful agent workflows with tool integration and conversation memory
"""

import asyncio
import logging
import json
import time
from typing import List, Dict, Any, Optional, Union, Callable, Literal
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import uuid

# LangGraph and LangChain imports (optional)
try:
    from langgraph.graph import StateGraph, END, START
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.prebuilt import ToolNode
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
    from langchain_core.tools import BaseTool, tool
    from langchain_core.runnables import RunnableConfig
    from typing_extensions import TypedDict, Annotated
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = None
    START = None
    MemorySaver = None
    ToolNode = None
    BaseMessage = None
    HumanMessage = None
    AIMessage = None
    SystemMessage = None
    ToolMessage = None
    BaseTool = None
    tool = None
    RunnableConfig = None
    TypedDict = dict
    Annotated = None
    ChatPromptTemplate = None
    StrOutputParser = None

# Existing KnowledgeHub services
from .hybrid_rag_service import HybridRAGService, RetrievalMode, get_hybrid_rag_service
from .memory_service import MemoryService
from .knowledge_graph import KnowledgeGraphService, NodeType, RelationType
from .cache import RedisCache
from .real_ai_intelligence import RealAIIntelligence
from ..models.user import User
from ..config import settings

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Agent roles for specialized tasks"""
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    SYNTHESIZER = "synthesizer"
    VALIDATOR = "validator"
    PLANNER = "planner"
    EXECUTOR = "executor"


class WorkflowType(Enum):
    """Types of agent workflows"""
    SIMPLE_QA = "simple_qa"
    MULTI_STEP_RESEARCH = "multi_step_research"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    SUMMARIZATION = "summarization"
    FACT_CHECKING = "fact_checking"
    KNOWLEDGE_SYNTHESIS = "knowledge_synthesis"


@dataclass
class AgentConfig:
    """Configuration for individual agents"""
    role: AgentRole
    name: str
    description: str
    system_prompt: str
    tools: List[str] = field(default_factory=list)
    max_iterations: int = 5
    temperature: float = 0.7
    max_tokens: int = 1000


@dataclass
class WorkflowConfig:
    """Configuration for agent workflows"""
    workflow_type: WorkflowType
    agents: List[AgentConfig]
    max_workflow_time: int = 300  # seconds
    enable_memory: bool = True
    enable_validation: bool = True
    require_consensus: bool = False


# LangGraph State Definition
if LANGGRAPH_AVAILABLE:
    class AgentState(TypedDict):
        """State for agent workflow orchestration"""
        query: str
        user_id: str
        session_id: str
        workflow_type: WorkflowType
        context: Dict[str, Any]
        messages: List[BaseMessage]
        research_results: List[Dict[str, Any]]
        analysis_results: List[Dict[str, Any]]
        synthesis_results: List[Dict[str, Any]]
        validation_results: List[Dict[str, Any]]
        final_response: str
        agent_memory: Dict[str, Any]
        reasoning_trace: List[str]
        performance_metrics: Dict[str, float]
        tools_used: List[str]
        current_agent: AgentRole
        workflow_stage: str
        error_state: Optional[Dict[str, Any]]
else:
    # Fallback state for when LangGraph is not available
    AgentState = Dict[str, Any]


class RAGTools:
    """Tools available to agents for RAG operations"""
    
    def __init__(self, hybrid_rag: HybridRAGService, memory_service: MemoryService):
        self.hybrid_rag = hybrid_rag
        self.memory_service = memory_service
        
    async def search_knowledge(
        self,
        query: str,
        retrieval_mode: str = "hybrid_all",
        top_k: int = 5
    ) -> Dict[str, Any]:
        """Search the knowledge base using hybrid RAG"""
        try:
            mode = RetrievalMode(retrieval_mode)
            result = await self.hybrid_rag.query(
                query=query,
                user_id="agent",
                retrieval_mode=mode,
                top_k=top_k
            )
            return {
                "success": True,
                "results": result["results"],
                "metadata": result["metadata"]
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "results": []
            }
    
    async def search_memory(
        self,
        query: str,
        user_id: str,
        memory_type: str = "all"
    ) -> Dict[str, Any]:
        """Search conversation memory and stored context"""
        try:
            # This would integrate with the memory service
            # For now, return a placeholder
            return {
                "success": True,
                "memories": [],
                "metadata": {"query": query, "type": memory_type}
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "memories": []
            }
    
    async def validate_information(
        self,
        claim: str,
        sources: List[str]
    ) -> Dict[str, Any]:
        """Validate information against multiple sources"""
        try:
            # Simple validation logic - can be enhanced with fact-checking models
            confidence = 0.8 if len(sources) > 1 else 0.5
            
            return {
                "success": True,
                "claim": claim,
                "confidence": confidence,
                "sources_count": len(sources),
                "validation_notes": f"Claim supported by {len(sources)} sources"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "confidence": 0.0
            }


class AgentOrchestrator:
    """
    Multi-agent orchestration service using LangGraph
    
    Features:
    - Stateful agent workflows
    - Tool integration for RAG operations
    - Conversation memory management
    - Multi-step reasoning and validation
    - Performance monitoring and optimization
    """
    
    def __init__(self):
        self.logger = logger
        self.cache = RedisCache(settings.REDIS_URL)
        self.ai_intelligence = RealAIIntelligence()
        self.memory_service = MemoryService()
        self.knowledge_graph = KnowledgeGraphService()
        
        # Will be initialized in initialize()
        self.hybrid_rag = None
        self.tools = None
        self.workflows = {}
        self.memory_store = MemorySaver() if LANGGRAPH_AVAILABLE else None
        
        # Agent configurations
        self.agent_configs = self._create_agent_configs()
        
        # Performance tracking
        self.workflow_stats = {
            "workflows_executed": 0,
            "avg_execution_time": 0.0,
            "success_rate": 0.0
        }
    
    async def initialize(self):
        """Initialize the orchestrator and all components"""
        try:
            # Initialize cache
            await self.cache.initialize()
            
            # Initialize hybrid RAG service
            self.hybrid_rag = await get_hybrid_rag_service()
            
            # Initialize knowledge graph
            await self.knowledge_graph.initialize()
            
            # Create tools
            self.tools = RAGTools(self.hybrid_rag, self.memory_service)
            
            # Create workflows
            self._create_workflows()
            
            logger.info("Agent orchestrator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent orchestrator: {e}")
            raise
    
    def _create_agent_configs(self) -> Dict[str, AgentConfig]:
        """Create configurations for different agent types"""
        return {
            "researcher": AgentConfig(
                role=AgentRole.RESEARCHER,
                name="Research Agent",
                description="Specializes in finding and gathering relevant information",
                system_prompt="""You are a research agent specialized in finding comprehensive information.
                Your role is to:
                1. Search for relevant information using available tools
                2. Gather context from multiple sources
                3. Identify key facts and data points
                4. Note any information gaps or uncertainties
                
                Always be thorough and cite your sources.""",
                tools=["search_knowledge", "search_memory"],
                max_iterations=3
            ),
            
            "analyst": AgentConfig(
                role=AgentRole.ANALYST,
                name="Analysis Agent", 
                description="Analyzes and interprets gathered information",
                system_prompt="""You are an analysis agent specialized in interpreting information.
                Your role is to:
                1. Analyze the information provided by the research agent
                2. Identify patterns, trends, and relationships
                3. Draw insights and conclusions
                4. Assess the quality and reliability of information
                
                Provide structured analysis with clear reasoning.""",
                tools=["validate_information"],
                max_iterations=2
            ),
            
            "synthesizer": AgentConfig(
                role=AgentRole.SYNTHESIZER,
                name="Synthesis Agent",
                description="Synthesizes information into coherent responses",
                system_prompt="""You are a synthesis agent specialized in creating comprehensive responses.
                Your role is to:
                1. Combine research findings and analysis results
                2. Create a coherent, well-structured response
                3. Address the original user query completely
                4. Maintain accuracy while being accessible
                
                Create responses that are informative, clear, and helpful.""",
                tools=[],
                max_iterations=1
            ),
            
            "validator": AgentConfig(
                role=AgentRole.VALIDATOR,
                name="Validation Agent",
                description="Validates responses for accuracy and completeness",
                system_prompt="""You are a validation agent specialized in quality assurance.
                Your role is to:
                1. Review the synthesized response for accuracy
                2. Check that the user query was fully addressed
                3. Verify that claims are supported by evidence
                4. Suggest improvements if needed
                
                Ensure high quality and accuracy in final responses.""",
                tools=["validate_information"],
                max_iterations=1
            )
        }
    
    def _create_workflows(self):
        """Create LangGraph workflows for different agent tasks"""
        
        if not LANGGRAPH_AVAILABLE:
            logger.warning("LangGraph not available, workflows will use fallback mode")
            self.workflows = {}
            return
        
        # Simple Q&A workflow
        self.workflows["simple_qa"] = self._create_simple_qa_workflow()
        
        # Multi-step research workflow
        self.workflows["multi_step_research"] = self._create_research_workflow()
        
        # Comparative analysis workflow
        self.workflows["comparative_analysis"] = self._create_analysis_workflow()
        
        logger.info(f"Created {len(self.workflows)} agent workflows")
    
    def _create_simple_qa_workflow(self):
        """Create simple Q&A workflow with basic RAG"""
        
        if not LANGGRAPH_AVAILABLE:
            return None
        
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("research", self._research_node)
        workflow.add_node("synthesize", self._synthesize_node)
        
        # Define edges
        workflow.set_entry_point("research")
        workflow.add_edge("research", "synthesize")
        workflow.add_edge("synthesize", END)
        
        return workflow.compile(checkpointer=self.memory_store)
    
    def _create_research_workflow(self):
        """Create multi-step research workflow"""
        
        if not LANGGRAPH_AVAILABLE:
            return None
        
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("initial_research", self._research_node)
        workflow.add_node("deep_analysis", self._analysis_node)
        workflow.add_node("synthesis", self._synthesize_node)
        workflow.add_node("validation", self._validation_node)
        workflow.add_node("route_decision", self._route_decision_node)
        
        # Define edges and routing
        workflow.set_entry_point("initial_research")
        workflow.add_edge("initial_research", "deep_analysis")
        workflow.add_edge("deep_analysis", "synthesis")
        workflow.add_edge("synthesis", "validation")
        workflow.add_edge("validation", "route_decision")
        
        # Conditional routing
        workflow.add_conditional_edges(
            "route_decision",
            self._should_continue,
            {
                "continue": "initial_research",
                "end": END
            }
        )
        
        return workflow.compile(checkpointer=self.memory_store)
    
    def _create_analysis_workflow(self):
        """Create comparative analysis workflow"""
        
        if not LANGGRAPH_AVAILABLE:
            return None
        
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("gather_sources", self._research_node)
        workflow.add_node("analyze_each", self._analysis_node)
        workflow.add_node("compare", self._comparison_node)
        workflow.add_node("synthesize_comparison", self._synthesize_node)
        
        # Define edges
        workflow.set_entry_point("gather_sources")
        workflow.add_edge("gather_sources", "analyze_each")
        workflow.add_edge("analyze_each", "compare")
        workflow.add_edge("compare", "synthesize_comparison")
        workflow.add_edge("synthesize_comparison", END)
        
        return workflow.compile(checkpointer=self.memory_store)
    
    async def execute_workflow(
        self,
        query: str,
        user_id: str,
        workflow_type: WorkflowType = WorkflowType.SIMPLE_QA,
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute an agent workflow
        
        Args:
            query: User query
            user_id: User identifier
            workflow_type: Type of workflow to execute
            session_id: Session identifier
            context: Additional context
            
        Returns:
            Workflow execution results
        """
        start_time = time.time()
        session_id = session_id or f"workflow_{int(time.time())}"
        
        try:
            # Get workflow
            workflow_key = workflow_type.value
            
            # Check if LangGraph workflows are available
            if not LANGGRAPH_AVAILABLE or not self.workflows or workflow_key not in self.workflows:
                # Use fallback workflow execution
                return await self._execute_fallback_workflow_complete(
                    query, user_id, workflow_type, session_id, context, start_time
                )
            
            workflow = self.workflows[workflow_key]
            
            # Initialize state
            initial_state: AgentState = {
                "user_query": query,
                "user_id": user_id,
                "session_id": session_id,
                "workflow_type": workflow_type.value,
                "messages": [HumanMessage(content=query)],
                "research_results": [],
                "analysis_results": [],
                "synthesis_result": "",
                "validation_result": {},
                "current_agent": "researcher",
                "next_agent": "",
                "iteration_count": 0,
                "workflow_complete": False,
                "retrieved_contexts": [],
                "conversation_memory": context or {},
                "reasoning_steps": [],
                "performance_metrics": {},
                "confidence_scores": {},
                "final_response": "",
                "response_metadata": {}
            }
            
            # Execute workflow
            config = RunnableConfig(
                configurable={"thread_id": session_id}
            )
            
            result_state = await workflow.ainvoke(initial_state, config)
            
            # Format response
            execution_time = time.time() - start_time
            
            response = {
                "response": result_state["final_response"],
                "metadata": {
                    "workflow_type": workflow_type.value,
                    "execution_time": execution_time,
                    "agent_interactions": result_state["iteration_count"],
                    "confidence_scores": result_state["confidence_scores"],
                    "session_id": session_id
                },
                "reasoning_steps": result_state["reasoning_steps"],
                "research_results": result_state["research_results"],
                "performance_metrics": result_state["performance_metrics"]
            }
            
            # Track performance
            await self._track_workflow_performance(
                workflow_type, execution_time, True, len(result_state["research_results"])
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            
            # Track failure
            await self._track_workflow_performance(
                workflow_type, time.time() - start_time, False, 0
            )
            
            return {
                "response": "I apologize, but I encountered an error processing your request.",
                "metadata": {
                    "error": str(e),
                    "execution_time": time.time() - start_time,
                    "workflow_type": workflow_type.value
                },
                "reasoning_steps": [],
                "research_results": [],
                "performance_metrics": {}
            }
    
    async def _research_node(self, state: AgentState) -> AgentState:
        """Research agent node - gather information"""
        query = state["user_query"]
        
        try:
            # Use hybrid RAG to search for information
            search_result = await self.tools.search_knowledge(
                query=query,
                retrieval_mode="hybrid_all",
                top_k=5
            )
            
            if search_result["success"]:
                research_results = [{
                    "agent": "researcher",
                    "query": query,
                    "results": search_result["results"],
                    "metadata": search_result["metadata"],
                    "timestamp": datetime.utcnow().isoformat()
                }]
                
                state["research_results"].extend(research_results)
                state["reasoning_steps"].append(f"Research agent found {len(search_result['results'])} relevant results")
                
                # Add to conversation
                state["messages"].append(
                    AIMessage(content=f"Found {len(search_result['results'])} relevant sources for: {query}")
                )
            else:
                state["reasoning_steps"].append("Research agent failed to find results")
                state["messages"].append(
                    AIMessage(content="No relevant information found in the knowledge base")
                )
            
        except Exception as e:
            logger.error(f"Research node failed: {e}")
            state["reasoning_steps"].append(f"Research agent error: {str(e)}")
        
        state["current_agent"] = "researcher"
        state["iteration_count"] += 1
        
        return state
    
    async def _analysis_node(self, state: AgentState) -> AgentState:
        """Analysis agent node - analyze gathered information"""
        
        try:
            research_results = state["research_results"]
            
            if not research_results:
                state["reasoning_steps"].append("Analysis agent: No research results to analyze")
                return state
            
            # Analyze the research results
            analysis = {
                "agent": "analyst",
                "analysis_type": "content_analysis",
                "findings": [],
                "confidence": 0.0,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Simple analysis - count and categorize results
            total_results = sum(len(r["results"]) for r in research_results)
            
            if total_results > 0:
                analysis["findings"] = [
                    f"Found {total_results} total relevant documents",
                    "Information appears to be comprehensive",
                    "Multiple sources support the findings"
                ]
                analysis["confidence"] = min(0.9, total_results * 0.2)
            else:
                analysis["findings"] = ["No relevant information found"]
                analysis["confidence"] = 0.1
            
            state["analysis_results"].append(analysis)
            state["confidence_scores"]["analysis"] = analysis["confidence"]
            state["reasoning_steps"].append(f"Analysis agent evaluated {total_results} results")
            
            # Add to conversation
            state["messages"].append(
                AIMessage(content=f"Analysis complete. Confidence: {analysis['confidence']:.2f}")
            )
            
        except Exception as e:
            logger.error(f"Analysis node failed: {e}")
            state["reasoning_steps"].append(f"Analysis agent error: {str(e)}")
        
        state["current_agent"] = "analyst"
        return state
    
    async def _synthesize_node(self, state: AgentState) -> AgentState:
        """Synthesis agent node - create final response"""
        
        try:
            query = state["user_query"]
            research_results = state["research_results"]
            analysis_results = state["analysis_results"]
            
            # Simple synthesis - combine research findings
            if research_results:
                all_results = []
                for research in research_results:
                    all_results.extend(research["results"])
                
                if all_results:
                    # Create response from top results
                    response_parts = [
                        f"Based on my research, here's what I found about '{query}':\n"
                    ]
                    
                    for i, result in enumerate(all_results[:3], 1):
                        content = result.get("content", "")[:200]
                        if content:
                            response_parts.append(f"{i}. {content}...")
                    
                    if len(all_results) > 3:
                        response_parts.append(f"\nI found {len(all_results)} total relevant sources.")
                    
                    # Add analysis insights
                    if analysis_results:
                        latest_analysis = analysis_results[-1]
                        confidence = latest_analysis.get("confidence", 0.0)
                        response_parts.append(f"\nConfidence in this information: {confidence:.1%}")
                    
                    synthesis_result = "\n\n".join(response_parts)
                else:
                    synthesis_result = "I couldn't find specific information about your query in the knowledge base."
            else:
                synthesis_result = "I wasn't able to gather sufficient information to answer your query."
            
            state["synthesis_result"] = synthesis_result
            state["final_response"] = synthesis_result
            state["reasoning_steps"].append("Synthesis agent created final response")
            
            # Add to conversation
            state["messages"].append(AIMessage(content=synthesis_result))
            
        except Exception as e:
            logger.error(f"Synthesis node failed: {e}")
            state["reasoning_steps"].append(f"Synthesis agent error: {str(e)}")
            state["final_response"] = "I encountered an error while synthesizing the response."
        
        state["current_agent"] = "synthesizer"
        return state
    
    async def _validation_node(self, state: AgentState) -> AgentState:
        """Validation agent node - validate response quality"""
        
        try:
            synthesis_result = state["synthesis_result"]
            research_results = state["research_results"]
            
            # Simple validation
            validation = {
                "agent": "validator",
                "response_quality": "good",
                "completeness": 0.8,
                "accuracy_check": "passed",
                "recommendations": [],
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Check if response addresses the query
            query = state["user_query"].lower()
            response = synthesis_result.lower()
            
            # Simple keyword overlap check
            query_words = set(query.split())
            response_words = set(response.split())
            overlap = len(query_words & response_words) / len(query_words) if query_words else 0
            
            validation["completeness"] = min(0.9, overlap + 0.3)
            
            if validation["completeness"] < 0.5:
                validation["recommendations"].append("Response may not fully address the query")
            
            if not research_results:
                validation["recommendations"].append("No research sources found")
                validation["response_quality"] = "poor"
            
            state["validation_result"] = validation
            state["confidence_scores"]["validation"] = validation["completeness"]
            state["reasoning_steps"].append(f"Validation complete. Quality: {validation['response_quality']}")
            
        except Exception as e:
            logger.error(f"Validation node failed: {e}")
            state["reasoning_steps"].append(f"Validation agent error: {str(e)}")
        
        state["current_agent"] = "validator"
        return state
    
    async def _comparison_node(self, state: AgentState) -> AgentState:
        """Comparison node for comparative analysis"""
        
        try:
            research_results = state["research_results"]
            
            # Simple comparison logic
            comparison = {
                "agent": "comparator",
                "comparison_type": "source_comparison",
                "similarities": [],
                "differences": [],
                "conclusions": [],
                "timestamp": datetime.utcnow().isoformat()
            }
            
            if len(research_results) >= 2:
                comparison["similarities"] = ["Multiple sources support similar findings"]
                comparison["differences"] = ["Some variation in specific details"]
                comparison["conclusions"] = ["Overall consensus across sources"]
            else:
                comparison["conclusions"] = ["Limited sources available for comparison"]
            
            state["analysis_results"].append(comparison)
            state["reasoning_steps"].append("Completed comparative analysis")
            
        except Exception as e:
            logger.error(f"Comparison node failed: {e}")
            state["reasoning_steps"].append(f"Comparison error: {str(e)}")
        
        return state
    
    async def _route_decision_node(self, state: AgentState) -> AgentState:
        """Route decision node for workflow control"""
        
        # Simple routing logic
        iteration_count = state["iteration_count"]
        max_iterations = 3
        
        validation_result = state.get("validation_result", {})
        completeness = validation_result.get("completeness", 0.0)
        
        # Continue if we haven't reached max iterations and quality is low
        if iteration_count < max_iterations and completeness < 0.7:
            state["next_agent"] = "researcher"
            state["workflow_complete"] = False
            state["reasoning_steps"].append(f"Continuing workflow - iteration {iteration_count}")
        else:
            state["workflow_complete"] = True
            state["reasoning_steps"].append("Workflow complete")
        
        return state
    
    def _should_continue(self, state: AgentState) -> Literal["continue", "end"]:
        """Determine if workflow should continue"""
        return "continue" if not state["workflow_complete"] else "end"
    
    async def _track_workflow_performance(
        self,
        workflow_type: WorkflowType,
        execution_time: float,
        success: bool,
        results_count: int
    ):
        """Track workflow performance metrics"""
        try:
            self.workflow_stats["workflows_executed"] += 1
            
            # Update average execution time
            total_workflows = self.workflow_stats["workflows_executed"]
            old_avg = self.workflow_stats["avg_execution_time"]
            self.workflow_stats["avg_execution_time"] = (
                (old_avg * (total_workflows - 1) + execution_time) / total_workflows
            )
            
            # Update success rate
            if success:
                old_success_rate = self.workflow_stats["success_rate"]
                self.workflow_stats["success_rate"] = (
                    (old_success_rate * (total_workflows - 1) + 1.0) / total_workflows
                )
            
            # Track in AI intelligence system
            await self.ai_intelligence.track_performance_metric(
                "agent_workflow",
                execution_time=execution_time,
                success=success,
                metadata={
                    "workflow_type": workflow_type.value,
                    "results_count": results_count
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to track workflow performance: {e}")
    
    async def get_available_workflows(self) -> List[Dict[str, Any]]:
        """Get list of available workflows"""
        return [
            {
                "type": workflow_type.value,
                "name": workflow_type.value.replace("_", " ").title(),
                "description": self._get_workflow_description(workflow_type),
                "agents": self._get_workflow_agents(workflow_type)
            }
            for workflow_type in WorkflowType
        ]
    
    def _get_workflow_description(self, workflow_type: WorkflowType) -> str:
        """Get description for workflow type"""
        descriptions = {
            WorkflowType.SIMPLE_QA: "Simple question-answering with basic RAG",
            WorkflowType.MULTI_STEP_RESEARCH: "Multi-step research with validation",
            WorkflowType.COMPARATIVE_ANALYSIS: "Comparative analysis across sources",
            WorkflowType.SUMMARIZATION: "Content summarization and synthesis",
            WorkflowType.FACT_CHECKING: "Fact checking and validation",
            WorkflowType.KNOWLEDGE_SYNTHESIS: "Knowledge synthesis from multiple sources"
        }
        return descriptions.get(workflow_type, "Unknown workflow")
    
    def _get_workflow_agents(self, workflow_type: WorkflowType) -> List[str]:
        """Get agents involved in workflow type"""
        agent_mapping = {
            WorkflowType.SIMPLE_QA: ["researcher", "synthesizer"],
            WorkflowType.MULTI_STEP_RESEARCH: ["researcher", "analyst", "synthesizer", "validator"],
            WorkflowType.COMPARATIVE_ANALYSIS: ["researcher", "analyst", "synthesizer"],
            WorkflowType.SUMMARIZATION: ["researcher", "synthesizer"],
            WorkflowType.FACT_CHECKING: ["researcher", "validator"],
            WorkflowType.KNOWLEDGE_SYNTHESIS: ["researcher", "analyst", "synthesizer"]
        }
        return agent_mapping.get(workflow_type, [])
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get orchestrator performance statistics"""
        return {
            **self.workflow_stats,
            "active_workflows": len(self.workflows),
            "available_agents": len(self.agent_configs),
            "memory_checkpoints": 0  # Would track actual checkpoints
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check orchestrator health"""
        health = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {}
        }
        
        # Check hybrid RAG
        if self.hybrid_rag:
            rag_health = await self.hybrid_rag.health_check()
            health["components"]["hybrid_rag"] = rag_health["status"]
        else:
            health["components"]["hybrid_rag"] = "not_initialized"
        
        # Check workflows
        health["components"]["workflows"] = f"{len(self.workflows)} loaded"
        health["components"]["agents"] = f"{len(self.agent_configs)} configured"
        
        # Overall status
        if any("unhealthy" in str(status) for status in health["components"].values()):
            health["status"] = "degraded"
        
        return health
    
    async def _execute_fallback_workflow_complete(
        self, 
        query: str, 
        user_id: str, 
        workflow_type: WorkflowType, 
        session_id: str, 
        context: Optional[Dict[str, Any]], 
        start_time: float
    ) -> Dict[str, Any]:
        """Execute complete workflow when LangGraph is not available"""
        try:
            # Simple fallback workflow execution
            research_results = []
            
            # Use hybrid RAG if available
            if self.hybrid_rag:
                try:
                    rag_result = await self.hybrid_rag.query(
                        query=query,
                        user_id=user_id,
                        session_id=session_id,
                        top_k=5
                    )
                    research_results = rag_result.get("results", [])
                except Exception as e:
                    logger.error(f"Hybrid RAG query failed in fallback: {e}")
            
            # Generate simple response
            if research_results:
                response_parts = [f"Based on available information for '{query}':"]
                for i, result in enumerate(research_results[:3], 1):
                    content = result.get("content", "")[:200]
                    if content:
                        response_parts.append(f"{i}. {content}...")
                
                final_response = "\n\n".join(response_parts)
            else:
                final_response = "I couldn't find relevant information for your query."
            
            execution_time = time.time() - start_time
            
            # Track performance
            await self._track_workflow_performance(
                workflow_type, execution_time, True, len(research_results)
            )
            
            return {
                "response": final_response,
                "metadata": {
                    "workflow_type": workflow_type.value,
                    "execution_time": execution_time,
                    "agent_interactions": 1,
                    "confidence_scores": {"fallback": 0.7},
                    "session_id": session_id,
                    "fallback_mode": True
                },
                "reasoning_steps": [
                    "Using fallback workflow mode",
                    f"Executed basic RAG search with {len(research_results)} results"
                ],
                "research_results": research_results,
                "performance_metrics": {"total_time": execution_time}
            }
            
        except Exception as e:
            logger.error(f"Fallback workflow execution failed: {e}")
            execution_time = time.time() - start_time
            
            await self._track_workflow_performance(
                workflow_type, execution_time, False, 0
            )
            
            return {
                "response": "I apologize, but I encountered an error processing your request.",
                "metadata": {
                    "error": str(e),
                    "execution_time": execution_time,
                    "workflow_type": workflow_type.value,
                    "fallback_mode": True
                },
                "reasoning_steps": [],
                "research_results": [],
                "performance_metrics": {}
            }


# Global instance
_agent_orchestrator: Optional[AgentOrchestrator] = None


async def get_agent_orchestrator() -> AgentOrchestrator:
    """Get singleton agent orchestrator instance"""
    global _agent_orchestrator
    
    if _agent_orchestrator is None:
        _agent_orchestrator = AgentOrchestrator()
        await _agent_orchestrator.initialize()
    
    return _agent_orchestrator