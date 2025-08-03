"""
Specialized Agents for Multi-Agent System
Each agent handles specific types of queries and tasks
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import asyncio

import logging
logger = logging.getLogger(__name__)
from ..rag.simple_rag_service import SimpleRAGService as LlamaIndexRAGService
from ..zep_memory import ZepMemoryService


class BaseAgent(ABC):
    """Base class for all specialized agents"""
    
    def __init__(self, rag_service: LlamaIndexRAGService):
        self.rag_service = rag_service
        self.logger = logger
    
    @abstractmethod
    async def execute(self, task: 'AgentTask') -> Any:
        """Execute the given task"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """Return agent capabilities"""
        pass
    
    async def query_rag(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Helper method to query RAG system"""
        return await self.rag_service.query(
            query_text=query,
            filters=filters,
            top_k=top_k
        )


class DocumentationAgent(BaseAgent):
    """Agent specialized in documentation queries"""
    
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "description": "Searches and analyzes technical documentation",
            "strengths": ["API references", "tutorials", "guides", "examples"],
            "sources": ["FastAPI", "React", "Django", "PostgreSQL", "Docker", 
                       "Kubernetes", "TypeScript", "Redis", "Elasticsearch", "Nginx"]
        }
    
    async def execute(self, task: 'AgentTask') -> Any:
        """Search documentation for the given query"""
        try:
            self.logger.info(f"DocumentationAgent executing: {task.description}")
            
            # Extract search parameters
            query = task.context.get("query", task.description)
            sources = task.context.get("sources", [])
            
            # Build filters for documentation sources
            filters = {"source_type": "documentation"}
            if sources:
                filters["source"] = {"$in": sources}
            
            # Search documentation
            results = await self.query_rag(
                query=query,
                filters=filters,
                top_k=task.context.get("top_k", 10)
            )
            
            # Format response
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "content": result.get("content", ""),
                    "source": result.get("metadata", {}).get("source", "Unknown"),
                    "url": result.get("metadata", {}).get("url", ""),
                    "relevance_score": result.get("score", 0.0)
                })
            
            return {
                "documentation_found": len(formatted_results) > 0,
                "results": formatted_results,
                "summary": self._summarize_docs(formatted_results, query)
            }
            
        except Exception as e:
            self.logger.error(f"DocumentationAgent error: {str(e)}")
            raise
    
    def _summarize_docs(self, results: List[Dict[str, Any]], query: str) -> str:
        """Create a summary of documentation findings"""
        if not results:
            return f"No documentation found for: {query}"
        
        sources = list(set(r["source"] for r in results[:5]))
        return f"Found relevant documentation in {', '.join(sources)} covering {query}"


class CodebaseAgent(BaseAgent):
    """Agent specialized in codebase analysis"""
    
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "description": "Searches and analyzes code repositories",
            "strengths": ["code patterns", "implementations", "dependencies", "architecture"],
            "languages": ["Python", "JavaScript", "TypeScript", "Go", "Java"]
        }
    
    async def execute(self, task: 'AgentTask') -> Any:
        """Search codebase for patterns and implementations"""
        try:
            self.logger.info(f"CodebaseAgent executing: {task.description}")
            
            query = task.context.get("query", task.description)
            language = task.context.get("language")
            pattern_type = task.context.get("pattern_type", "implementation")
            
            # Build filters for code search
            filters = {"source_type": "code"}
            if language:
                filters["language"] = language
            
            # Search for code patterns
            results = await self.query_rag(
                query=f"{pattern_type} {query}",
                filters=filters,
                top_k=task.context.get("top_k", 15)
            )
            
            # Analyze code patterns
            patterns = self._analyze_patterns(results)
            
            return {
                "code_found": len(results) > 0,
                "patterns": patterns,
                "implementations": self._extract_implementations(results),
                "languages": list(set(
                    r.get("metadata", {}).get("language", "Unknown") 
                    for r in results
                ))
            }
            
        except Exception as e:
            self.logger.error(f"CodebaseAgent error: {str(e)}")
            raise
    
    def _analyze_patterns(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze code patterns from search results"""
        patterns = []
        
        # Group by pattern type (simplified for now)
        pattern_groups = {
            "functions": [],
            "classes": [],
            "imports": [],
            "configurations": []
        }
        
        for result in results:
            content = result.get("content", "")
            if "def " in content or "function " in content:
                pattern_groups["functions"].append(result)
            elif "class " in content:
                pattern_groups["classes"].append(result)
            elif "import " in content or "require(" in content:
                pattern_groups["imports"].append(result)
            elif "config" in content.lower():
                pattern_groups["configurations"].append(result)
        
        for pattern_type, items in pattern_groups.items():
            if items:
                patterns.append({
                    "type": pattern_type,
                    "count": len(items),
                    "examples": items[:3]  # Top 3 examples
                })
        
        return patterns
    
    def _extract_implementations(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract actual implementation examples"""
        implementations = []
        
        for result in results[:5]:  # Top 5 implementations
            implementations.append({
                "code": result.get("content", ""),
                "file": result.get("metadata", {}).get("file_path", "Unknown"),
                "language": result.get("metadata", {}).get("language", "Unknown"),
                "context": result.get("metadata", {}).get("context", "")
            })
        
        return implementations


class PerformanceAgent(BaseAgent):
    """Agent specialized in performance optimization"""
    
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "description": "Analyzes performance patterns and optimizations",
            "strengths": ["benchmarks", "optimization techniques", "profiling", "caching"],
            "metrics": ["latency", "throughput", "memory usage", "CPU usage"]
        }
    
    async def execute(self, task: 'AgentTask') -> Any:
        """Analyze performance-related queries"""
        try:
            self.logger.info(f"PerformanceAgent executing: {task.description}")
            
            query = task.context.get("query", task.description)
            metric_type = task.context.get("metric_type", "general")
            
            # Search for performance-related content
            perf_query = f"performance optimization {metric_type} {query}"
            results = await self.query_rag(
                query=perf_query,
                filters={"$or": [
                    {"content": {"$contains": "performance"}},
                    {"content": {"$contains": "optimization"}},
                    {"content": {"$contains": "benchmark"}}
                ]},
                top_k=10
            )
            
            # Analyze performance patterns
            recommendations = self._generate_recommendations(results, query)
            benchmarks = self._extract_benchmarks(results)
            
            return {
                "performance_insights": len(results) > 0,
                "recommendations": recommendations,
                "benchmarks": benchmarks,
                "optimization_techniques": self._extract_techniques(results)
            }
            
        except Exception as e:
            self.logger.error(f"PerformanceAgent error: {str(e)}")
            raise
    
    def _generate_recommendations(
        self, 
        results: List[Dict[str, Any]], 
        query: str
    ) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        # Common performance patterns
        patterns = {
            "caching": "Implement caching to reduce repeated computations",
            "indexing": "Add database indexes for frequently queried fields",
            "async": "Use asynchronous operations for I/O-bound tasks",
            "batching": "Batch operations to reduce overhead",
            "pooling": "Use connection pooling for database connections"
        }
        
        for pattern, recommendation in patterns.items():
            if any(pattern in r.get("content", "").lower() for r in results):
                recommendations.append(recommendation)
        
        return recommendations[:5]  # Top 5 recommendations
    
    def _extract_benchmarks(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract benchmark data from results"""
        benchmarks = []
        
        for result in results:
            content = result.get("content", "")
            # Look for numeric performance data (simplified)
            if any(keyword in content.lower() for keyword in ["ms", "seconds", "throughput", "qps"]):
                benchmarks.append({
                    "description": content[:200] + "...",
                    "source": result.get("metadata", {}).get("source", "Unknown")
                })
        
        return benchmarks[:3]
    
    def _extract_techniques(self, results: List[Dict[str, Any]]) -> List[str]:
        """Extract optimization techniques"""
        techniques = set()
        
        technique_keywords = [
            "caching", "memoization", "lazy loading", "connection pooling",
            "query optimization", "indexing", "compression", "CDN",
            "load balancing", "horizontal scaling", "vertical scaling",
            "async/await", "parallel processing", "batch processing"
        ]
        
        for result in results:
            content = result.get("content", "").lower()
            for keyword in technique_keywords:
                if keyword in content:
                    techniques.add(keyword)
        
        return list(techniques)[:10]


class StyleGuideAgent(BaseAgent):
    """Agent specialized in code style and best practices"""
    
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "description": "Analyzes code style and best practices",
            "strengths": ["coding standards", "design patterns", "conventions", "linting"],
            "frameworks": ["PEP8", "ESLint", "Prettier", "Black", "Clean Code"]
        }
    
    async def execute(self, task: 'AgentTask') -> Any:
        """Check style guide compliance and best practices"""
        try:
            self.logger.info(f"StyleGuideAgent executing: {task.description}")
            
            query = task.context.get("query", task.description)
            language = task.context.get("language", "python")
            
            # Search for style guides and best practices
            style_query = f"{language} style guide best practices {query}"
            results = await self.query_rag(
                query=style_query,
                filters={"$or": [
                    {"content": {"$contains": "style"}},
                    {"content": {"$contains": "convention"}},
                    {"content": {"$contains": "best practice"}}
                ]},
                top_k=8
            )
            
            return {
                "style_guidelines": self._extract_guidelines(results, language),
                "best_practices": self._extract_best_practices(results),
                "common_issues": self._identify_common_issues(results, query),
                "tools": self._recommend_tools(language)
            }
            
        except Exception as e:
            self.logger.error(f"StyleGuideAgent error: {str(e)}")
            raise
    
    def _extract_guidelines(
        self, 
        results: List[Dict[str, Any]], 
        language: str
    ) -> List[Dict[str, str]]:
        """Extract style guidelines"""
        guidelines = []
        
        # Language-specific guidelines
        guide_map = {
            "python": ["PEP 8", "Black formatting", "Type hints"],
            "javascript": ["ESLint", "Prettier", "Airbnb style guide"],
            "typescript": ["TSLint", "Prettier", "Microsoft style guide"],
            "go": ["gofmt", "golint", "Effective Go"],
            "java": ["Google Java Style", "Checkstyle", "SonarLint"]
        }
        
        for guide in guide_map.get(language.lower(), []):
            guidelines.append({
                "name": guide,
                "description": f"Follow {guide} conventions for {language}"
            })
        
        return guidelines
    
    def _extract_best_practices(self, results: List[Dict[str, Any]]) -> List[str]:
        """Extract general best practices"""
        practices = [
            "Use descriptive variable and function names",
            "Keep functions small and focused",
            "Write self-documenting code",
            "Follow DRY (Don't Repeat Yourself) principle",
            "Handle errors gracefully",
            "Write unit tests for critical functions",
            "Use version control effectively",
            "Document complex logic"
        ]
        
        # Filter based on results
        relevant_practices = []
        for practice in practices:
            if any(
                any(word in r.get("content", "").lower() 
                    for word in practice.lower().split())
                for r in results
            ):
                relevant_practices.append(practice)
        
        return relevant_practices[:5]
    
    def _identify_common_issues(
        self, 
        results: List[Dict[str, Any]], 
        query: str
    ) -> List[str]:
        """Identify common style issues"""
        issues = []
        
        issue_patterns = {
            "naming": "Inconsistent naming conventions",
            "indentation": "Mixed indentation (tabs vs spaces)",
            "imports": "Unorganized imports",
            "comments": "Lack of documentation",
            "complexity": "High cyclomatic complexity",
            "duplication": "Code duplication"
        }
        
        for pattern, issue in issue_patterns.items():
            if pattern in query.lower() or any(
                pattern in r.get("content", "").lower() for r in results
            ):
                issues.append(issue)
        
        return issues[:4]
    
    def _recommend_tools(self, language: str) -> List[Dict[str, str]]:
        """Recommend style checking tools"""
        tools_map = {
            "python": [
                {"name": "Black", "purpose": "Code formatting"},
                {"name": "Flake8", "purpose": "Style checking"},
                {"name": "mypy", "purpose": "Type checking"}
            ],
            "javascript": [
                {"name": "ESLint", "purpose": "Linting"},
                {"name": "Prettier", "purpose": "Code formatting"},
                {"name": "JSDoc", "purpose": "Documentation"}
            ],
            "typescript": [
                {"name": "TSLint", "purpose": "Linting"},
                {"name": "Prettier", "purpose": "Code formatting"},
                {"name": "TypeDoc", "purpose": "Documentation"}
            ]
        }
        
        return tools_map.get(language.lower(), [])


class TestingAgent(BaseAgent):
    """Agent specialized in testing strategies"""
    
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "description": "Provides testing strategies and examples",
            "strengths": ["unit testing", "integration testing", "E2E testing", "TDD"],
            "frameworks": ["pytest", "Jest", "Mocha", "JUnit", "Selenium"]
        }
    
    async def execute(self, task: 'AgentTask') -> Any:
        """Provide testing recommendations and examples"""
        try:
            self.logger.info(f"TestingAgent executing: {task.description}")
            
            query = task.context.get("query", task.description)
            test_type = task.context.get("test_type", "unit")
            language = task.context.get("language", "python")
            
            # Search for testing patterns
            test_query = f"{test_type} testing {language} {query}"
            results = await self.query_rag(
                query=test_query,
                filters={"$or": [
                    {"content": {"$contains": "test"}},
                    {"content": {"$contains": "assert"}},
                    {"content": {"$contains": "mock"}}
                ]},
                top_k=10
            )
            
            return {
                "testing_strategies": self._generate_strategies(test_type, language),
                "examples": self._extract_test_examples(results),
                "frameworks": self._recommend_frameworks(language, test_type),
                "best_practices": self._testing_best_practices(test_type)
            }
            
        except Exception as e:
            self.logger.error(f"TestingAgent error: {str(e)}")
            raise
    
    def _generate_strategies(self, test_type: str, language: str) -> List[Dict[str, str]]:
        """Generate testing strategies"""
        strategies = []
        
        if test_type == "unit":
            strategies = [
                {"name": "Test isolation", "description": "Test individual functions in isolation"},
                {"name": "Mock dependencies", "description": "Use mocks for external dependencies"},
                {"name": "Edge cases", "description": "Test boundary conditions and edge cases"}
            ]
        elif test_type == "integration":
            strategies = [
                {"name": "Test interactions", "description": "Test component interactions"},
                {"name": "Real dependencies", "description": "Use real databases/services when possible"},
                {"name": "Data consistency", "description": "Verify data flow between components"}
            ]
        elif test_type == "e2e":
            strategies = [
                {"name": "User journeys", "description": "Test complete user workflows"},
                {"name": "Browser automation", "description": "Use tools like Selenium or Playwright"},
                {"name": "Environment parity", "description": "Test in production-like environment"}
            ]
        
        return strategies
    
    def _extract_test_examples(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract test examples from results"""
        examples = []
        
        for result in results[:5]:
            content = result.get("content", "")
            if any(keyword in content for keyword in ["def test_", "it(", "describe(", "@Test"]):
                examples.append({
                    "code": content,
                    "type": self._identify_test_type(content),
                    "framework": self._identify_framework(content)
                })
        
        return examples
    
    def _identify_test_type(self, content: str) -> str:
        """Identify the type of test from content"""
        if "mock" in content.lower():
            return "unit"
        elif "integration" in content.lower():
            return "integration"
        elif "browser" in content.lower() or "selenium" in content.lower():
            return "e2e"
        return "unit"
    
    def _identify_framework(self, content: str) -> str:
        """Identify testing framework from content"""
        frameworks = {
            "pytest": ["pytest", "def test_"],
            "jest": ["jest", "describe(", "it("],
            "mocha": ["mocha", "describe(", "it("],
            "junit": ["junit", "@Test"],
            "unittest": ["unittest", "TestCase"]
        }
        
        for framework, keywords in frameworks.items():
            if any(keyword in content for keyword in keywords):
                return framework
        
        return "unknown"
    
    def _recommend_frameworks(self, language: str, test_type: str) -> List[Dict[str, str]]:
        """Recommend testing frameworks"""
        frameworks_map = {
            "python": {
                "unit": [{"name": "pytest", "description": "Modern Python testing"}],
                "integration": [{"name": "pytest + fixtures", "description": "Integration testing"}],
                "e2e": [{"name": "Playwright", "description": "Browser automation"}]
            },
            "javascript": {
                "unit": [{"name": "Jest", "description": "JavaScript testing"}],
                "integration": [{"name": "Mocha + Chai", "description": "Flexible testing"}],
                "e2e": [{"name": "Cypress", "description": "Modern E2E testing"}]
            }
        }
        
        return frameworks_map.get(language, {}).get(test_type, [])
    
    def _testing_best_practices(self, test_type: str) -> List[str]:
        """Testing best practices"""
        practices = {
            "unit": [
                "Keep tests fast and isolated",
                "One assertion per test",
                "Use descriptive test names",
                "Follow AAA pattern (Arrange, Act, Assert)"
            ],
            "integration": [
                "Use test databases",
                "Clean up test data",
                "Test error scenarios",
                "Verify data persistence"
            ],
            "e2e": [
                "Use page object pattern",
                "Handle async operations",
                "Test on multiple browsers",
                "Keep tests maintainable"
            ]
        }
        
        return practices.get(test_type, practices["unit"])


class SynthesisAgent(BaseAgent):
    """Agent that synthesizes results from other agents"""
    
    def __init__(
        self, 
        rag_service: LlamaIndexRAGService,
        zep_service: Optional[ZepMemoryService] = None
    ):
        super().__init__(rag_service)
        self.zep_service = zep_service
    
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "description": "Synthesizes and combines results from multiple agents",
            "strengths": ["result aggregation", "conflict resolution", "summary generation"],
            "output_formats": ["structured", "narrative", "recommendations"]
        }
    
    async def execute(self, task: 'AgentTask') -> Any:
        """Synthesize results from multiple agents"""
        try:
            self.logger.info(f"SynthesisAgent executing: {task.description}")
            
            original_query = task.context.get("original_query", "")
            agent_results = task.context.get("agent_results", {})
            output_format = task.context.get("output_format", "structured")
            
            # Get memory context if available
            memory_context = None
            if self.zep_service and task.context.get("session_id"):
                memory_context = await self._get_memory_context(
                    task.context["session_id"],
                    task.context.get("user_id")
                )
            
            # Synthesize based on format
            if output_format == "narrative":
                return self._create_narrative_response(
                    original_query, agent_results, memory_context
                )
            elif output_format == "recommendations":
                return self._create_recommendations(
                    original_query, agent_results, memory_context
                )
            else:
                return self._create_structured_response(
                    original_query, agent_results, memory_context
                )
                
        except Exception as e:
            self.logger.error(f"SynthesisAgent error: {str(e)}")
            raise
    
    async def _get_memory_context(
        self, 
        session_id: str, 
        user_id: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Get relevant memory context"""
        try:
            if not self.zep_service:
                return None
            
            # Get recent memory
            memory = await self.zep_service.get_memory(
                session_id=session_id,
                limit=5
            )
            
            return {
                "recent_queries": [m.get("content") for m in memory.get("messages", [])],
                "session_summary": memory.get("summary", "")
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get memory context: {str(e)}")
            return None
    
    def _create_structured_response(
        self,
        query: str,
        agent_results: Dict[str, Any],
        memory_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create a structured synthesis"""
        response = {
            "query": query,
            "summary": self._generate_summary(agent_results),
            "key_findings": self._extract_key_findings(agent_results),
            "recommendations": self._extract_recommendations(agent_results),
            "sources": self._aggregate_sources(agent_results),
            "confidence_score": self._calculate_confidence(agent_results)
        }
        
        if memory_context:
            response["context_note"] = "Previous queries considered in this response"
        
        return response
    
    def _create_narrative_response(
        self,
        query: str,
        agent_results: Dict[str, Any],
        memory_context: Optional[Dict[str, Any]]
    ) -> str:
        """Create a narrative synthesis"""
        sections = []
        
        # Introduction
        sections.append(f"Based on the analysis of '{query}', here's what I found:\n")
        
        # Main findings
        for task_id, result in agent_results.items():
            if result.get("result"):
                agent_name = result.get("agent", "Unknown")
                sections.append(f"\n{agent_name} Analysis:")
                sections.append(self._format_agent_result(result))
        
        # Conclusion
        sections.append("\nIn summary:")
        sections.append(self._generate_summary(agent_results))
        
        return "\n".join(sections)
    
    def _create_recommendations(
        self,
        query: str,
        agent_results: Dict[str, Any],
        memory_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create actionable recommendations"""
        all_recommendations = []
        
        # Extract recommendations from each agent
        for task_id, result in agent_results.items():
            agent_data = result.get("result", {})
            
            # Look for recommendations in different formats
            if "recommendations" in agent_data:
                all_recommendations.extend(agent_data["recommendations"])
            if "best_practices" in agent_data:
                all_recommendations.extend(agent_data["best_practices"])
            if "optimization_techniques" in agent_data:
                all_recommendations.extend(agent_data["optimization_techniques"])
        
        # Deduplicate and prioritize
        unique_recommendations = list(set(all_recommendations))
        
        return {
            "query": query,
            "recommendations": unique_recommendations[:10],  # Top 10
            "implementation_priority": self._prioritize_recommendations(unique_recommendations),
            "next_steps": self._generate_next_steps(unique_recommendations)
        }
    
    def _generate_summary(self, agent_results: Dict[str, Any]) -> str:
        """Generate a summary from all agent results"""
        key_points = []
        
        for task_id, result in agent_results.items():
            if result.get("result"):
                agent_name = result.get("agent", "Unknown")
                task_type = result.get("type", "general")
                
                # Extract main finding from each agent
                agent_data = result.get("result", {})
                if task_type == "documentation" and agent_data.get("documentation_found"):
                    key_points.append(f"Found relevant documentation in {len(agent_data.get('results', []))} sources")
                elif task_type == "code_search" and agent_data.get("code_found"):
                    key_points.append(f"Identified {len(agent_data.get('patterns', []))} code patterns")
                elif task_type == "performance" and agent_data.get("recommendations"):
                    key_points.append(f"Generated {len(agent_data.get('recommendations', []))} performance recommendations")
        
        return ". ".join(key_points) if key_points else "Analysis complete."
    
    def _extract_key_findings(self, agent_results: Dict[str, Any]) -> List[str]:
        """Extract key findings from all agents"""
        findings = []
        
        for task_id, result in agent_results.items():
            agent_data = result.get("result", {})
            
            # Extract findings based on agent type
            if "documentation_found" in agent_data and agent_data["documentation_found"]:
                findings.append("Relevant documentation available")
            if "patterns" in agent_data and agent_data["patterns"]:
                findings.append(f"Found {len(agent_data['patterns'])} code patterns")
            if "performance_insights" in agent_data and agent_data["performance_insights"]:
                findings.append("Performance optimization opportunities identified")
            if "style_guidelines" in agent_data:
                findings.append("Style guide recommendations available")
            if "testing_strategies" in agent_data:
                findings.append("Testing strategies provided")
        
        return findings
    
    def _extract_recommendations(self, agent_results: Dict[str, Any]) -> List[str]:
        """Extract all recommendations"""
        recommendations = []
        
        for task_id, result in agent_results.items():
            agent_data = result.get("result", {})
            if "recommendations" in agent_data:
                recommendations.extend(agent_data["recommendations"])
        
        return list(set(recommendations))[:5]  # Top 5 unique
    
    def _aggregate_sources(self, agent_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Aggregate all sources used"""
        sources = []
        seen = set()
        
        for task_id, result in agent_results.items():
            agent_data = result.get("result", {})
            
            # Extract sources from documentation agent
            if "results" in agent_data:
                for doc in agent_data["results"]:
                    source_key = (doc.get("source"), doc.get("url"))
                    if source_key not in seen:
                        seen.add(source_key)
                        sources.append({
                            "source": doc.get("source"),
                            "url": doc.get("url"),
                            "type": "documentation"
                        })
        
        return sources
    
    def _calculate_confidence(self, agent_results: Dict[str, Any]) -> float:
        """Calculate overall confidence score"""
        scores = []
        
        for task_id, result in agent_results.items():
            if "error" not in result:
                # Base confidence on whether results were found
                agent_data = result.get("result", {})
                if any(agent_data.get(key, False) for key in 
                       ["documentation_found", "code_found", "performance_insights"]):
                    scores.append(0.8)
                else:
                    scores.append(0.4)
            else:
                scores.append(0.2)
        
        return sum(scores) / len(scores) if scores else 0.5
    
    def _format_agent_result(self, result: Dict[str, Any]) -> str:
        """Format individual agent result for narrative"""
        agent_data = result.get("result", {})
        task_type = result.get("type", "general")
        
        if task_type == "documentation":
            return f"Found {len(agent_data.get('results', []))} relevant documentation sources."
        elif task_type == "code_search":
            patterns = agent_data.get('patterns', [])
            return f"Identified {len(patterns)} code patterns across {len(agent_data.get('languages', []))} languages."
        elif task_type == "performance":
            return f"Generated {len(agent_data.get('recommendations', []))} performance recommendations."
        elif task_type == "style_check":
            return f"Provided {len(agent_data.get('style_guidelines', []))} style guidelines."
        elif task_type == "testing":
            return f"Suggested {len(agent_data.get('testing_strategies', []))} testing strategies."
        
        return "Analysis completed."
    
    def _prioritize_recommendations(self, recommendations: List[str]) -> List[Dict[str, Any]]:
        """Prioritize recommendations by impact"""
        # Simple prioritization based on keywords
        high_priority_keywords = ["security", "performance", "error", "critical"]
        medium_priority_keywords = ["optimization", "best practice", "improve"]
        
        prioritized = []
        
        for rec in recommendations:
            rec_lower = rec.lower()
            if any(keyword in rec_lower for keyword in high_priority_keywords):
                priority = "high"
            elif any(keyword in rec_lower for keyword in medium_priority_keywords):
                priority = "medium"
            else:
                priority = "low"
            
            prioritized.append({
                "recommendation": rec,
                "priority": priority
            })
        
        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        prioritized.sort(key=lambda x: priority_order[x["priority"]])
        
        return prioritized[:10]
    
    def _generate_next_steps(self, recommendations: List[str]) -> List[str]:
        """Generate actionable next steps"""
        next_steps = []
        
        # Map recommendations to actions
        action_map = {
            "cache": "Implement Redis caching for frequently accessed data",
            "index": "Add database indexes to slow queries",
            "test": "Write unit tests for critical functions",
            "document": "Add documentation for public APIs",
            "refactor": "Refactor complex functions into smaller units"
        }
        
        for rec in recommendations[:5]:
            rec_lower = rec.lower()
            for keyword, action in action_map.items():
                if keyword in rec_lower:
                    next_steps.append(action)
                    break
        
        if not next_steps:
            next_steps = ["Review the recommendations above", "Prioritize based on your needs"]
        
        return next_steps