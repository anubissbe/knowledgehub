#!/usr/bin/env python3
"""
KnowledgeHub Performance Analysis Tool
Identifies duplicate imports, similar services, and optimization opportunities.
"""

import os
import ast
import re
import json
from collections import defaultdict, Counter
from pathlib import Path
import subprocess

class PerformanceAnalyzer:
    def __init__(self, base_path="/opt/projects/knowledgehub"):
        self.base_path = Path(base_path)
        self.results = {
            "duplicate_imports": defaultdict(list),
            "similar_services": [],
            "middleware_duplicates": [],
            "circular_dependencies": [],
            "large_files": [],
            "import_statistics": {},
            "optimization_opportunities": []
        }
    
    def analyze_imports(self):
        """Analyze import patterns across Python files"""
        import_tracker = defaultdict(list)
        
        for py_file in self.base_path.glob("**/*.py"):
            if py_file.name in ["__pycache__", ".git"]:
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Track file size
                file_size = len(content)
                if file_size > 50000:  # Files over 50KB
                    self.results["large_files"].append({
                        "file": str(py_file.relative_to(self.base_path)),
                        "size": file_size
                    })
                
                # Extract imports
                try:
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                import_tracker[alias.name].append(str(py_file.relative_to(self.base_path)))
                        elif isinstance(node, ast.ImportFrom):
                            module = node.module or ""
                            for alias in node.names:
                                full_import = f"{module}.{alias.name}" if module else alias.name
                                import_tracker[full_import].append(str(py_file.relative_to(self.base_path)))
                except SyntaxError:
                    continue
                    
            except (UnicodeDecodeError, IOError):
                continue
        
        # Find duplicates
        for import_name, files in import_tracker.items():
            if len(files) > 10:  # Threshold for considering duplicates
                self.results["duplicate_imports"][import_name] = files
        
        # Import statistics
        self.results["import_statistics"] = {
            "total_unique_imports": len(import_tracker),
            "most_common_imports": dict(Counter({k: len(v) for k, v in import_tracker.items()}).most_common(20))
        }
    
    def analyze_similar_services(self):
        """Find services with similar functionality that could be consolidated"""
        service_patterns = {
            "memory": ["memory", "storage", "cache", "persist"],
            "auth": ["auth", "login", "session", "token", "rbac"],
            "ai": ["ai", "ml", "intelligence", "semantic", "embedding", "rag"],
            "analytics": ["analytics", "metrics", "monitoring", "tracking"],
            "workflow": ["workflow", "task", "job", "pipeline", "orchestr"]
        }
        
        services_dir = self.base_path / "api" / "services"
        if services_dir.exists():
            service_files = list(services_dir.glob("*.py"))
            
            for category, keywords in service_patterns.items():
                matching_services = []
                for service_file in service_files:
                    if any(keyword in service_file.name.lower() for keyword in keywords):
                        matching_services.append(str(service_file.relative_to(self.base_path)))
                
                if len(matching_services) > 3:  # More than 3 similar services
                    self.results["similar_services"].append({
                        "category": category,
                        "files": matching_services,
                        "count": len(matching_services)
                    })
    
    def analyze_middleware_duplicates(self):
        """Find duplicate middleware implementations"""
        middleware_dir = self.base_path / "api" / "middleware"
        if middleware_dir.exists():
            middleware_functions = defaultdict(list)
            
            for middleware_file in middleware_dir.glob("*.py"):
                try:
                    with open(middleware_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # Look for function definitions
                    function_pattern = r'def\s+(\w+)\s*\('
                    functions = re.findall(function_pattern, content)
                    
                    for func in functions:
                        middleware_functions[func].append(str(middleware_file.relative_to(self.base_path)))
                        
                except (UnicodeDecodeError, IOError):
                    continue
            
            # Find duplicate functions
            for func_name, files in middleware_functions.items():
                if len(files) > 1:
                    self.results["middleware_duplicates"].append({
                        "function": func_name,
                        "files": files
                    })
    
    def check_circular_dependencies(self):
        """Check for potential circular dependencies using simple heuristics"""
        # This is a simplified check - real circular dependency detection requires more complex analysis
        import_graph = defaultdict(set)
        
        for py_file in self.base_path.glob("api/**/*.py"):
            if py_file.name.startswith("__"):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Look for internal imports
                internal_imports = re.findall(r'from\s+(?:api\.|\.)\S+\s+import', content)
                internal_imports.extend(re.findall(r'import\s+(?:api\.)\S+', content))
                
                file_module = str(py_file.relative_to(self.base_path)).replace('/', '.').replace('.py', '')
                
                for imp in internal_imports:
                    # Extract module name (simplified)
                    module_match = re.search(r'(?:from\s+|import\s+)((?:api\.)?[\w\.]+)', imp)
                    if module_match:
                        imported_module = module_match.group(1)
                        import_graph[file_module].add(imported_module)
                        
            except (UnicodeDecodeError, IOError):
                continue
        
        # Simple cycle detection (not comprehensive)
        for module, imports in import_graph.items():
            for imported in imports:
                if imported in import_graph and module in import_graph[imported]:
                    self.results["circular_dependencies"].append({
                        "module_a": module,
                        "module_b": imported
                    })
    
    def generate_optimization_opportunities(self):
        """Generate actionable optimization recommendations"""
        opportunities = []
        
        # Large import consolidation
        fastapi_imports = self.results["duplicate_imports"].get("fastapi", [])
        if len(fastapi_imports) > 50:
            opportunities.append({
                "type": "import_consolidation",
                "priority": "high",
                "description": f"Create shared FastAPI import module - used in {len(fastapi_imports)} files",
                "files_affected": len(fastapi_imports),
                "estimated_savings": "15-25% reduction in import overhead"
            })
        
        # Service consolidation
        for similar_group in self.results["similar_services"]:
            if similar_group["count"] > 5:
                opportunities.append({
                    "type": "service_consolidation",
                    "priority": "medium",
                    "description": f"Consolidate {similar_group['count']} {similar_group['category']} services",
                    "files_affected": similar_group["count"],
                    "estimated_savings": "30-40% reduction in code duplication"
                })
        
        # Middleware consolidation
        if len(self.results["middleware_duplicates"]) > 5:
            opportunities.append({
                "type": "middleware_consolidation",
                "priority": "medium",
                "description": f"Consolidate {len(self.results['middleware_duplicates'])} duplicate middleware functions",
                "files_affected": sum(len(dup["files"]) for dup in self.results["middleware_duplicates"]),
                "estimated_savings": "20-30% reduction in middleware overhead"
            })
        
        # Large file optimization
        if len(self.results["large_files"]) > 10:
            opportunities.append({
                "type": "file_splitting",
                "priority": "low",
                "description": f"Split {len(self.results['large_files'])} large files (>50KB)",
                "files_affected": len(self.results["large_files"]),
                "estimated_savings": "10-15% improvement in load times"
            })
        
        self.results["optimization_opportunities"] = opportunities
    
    def run_full_analysis(self):
        """Run complete performance analysis"""
        print("🔍 Starting performance analysis...")
        
        print("  📦 Analyzing imports...")
        self.analyze_imports()
        
        print("  🔧 Analyzing similar services...")
        self.analyze_similar_services()
        
        print("  🛡️  Analyzing middleware duplicates...")
        self.analyze_middleware_duplicates()
        
        print("  🔄 Checking circular dependencies...")
        self.check_circular_dependencies()
        
        print("  💡 Generating optimization opportunities...")
        self.generate_optimization_opportunities()
        
        return self.results
    
    def print_report(self):
        """Print analysis report"""
        print("\n" + "="*80)
        print("🚀 KNOWLEDGEHUB PERFORMANCE ANALYSIS REPORT")
        print("="*80)
        
        # Import Statistics
        print(f"\n📊 IMPORT STATISTICS:")
        print(f"  Total unique imports: {self.results['import_statistics']['total_unique_imports']}")
        print(f"  Most duplicated imports:")
        for imp, count in list(self.results['import_statistics']['most_common_imports'].items())[:10]:
            print(f"    • {imp}: {count} files")
        
        # Similar Services
        print(f"\n🔧 SIMILAR SERVICES:")
        for group in self.results["similar_services"]:
            print(f"  • {group['category'].upper()}: {group['count']} files")
            for file in group['files'][:3]:  # Show first 3
                print(f"    - {file}")
            if len(group['files']) > 3:
                print(f"    ... and {len(group['files']) - 3} more")
        
        # Middleware Duplicates
        print(f"\n🛡️  MIDDLEWARE DUPLICATES:")
        for dup in self.results["middleware_duplicates"][:10]:  # Top 10
            print(f"  • {dup['function']}: {len(dup['files'])} files")
        
        # Large Files
        print(f"\n📄 LARGE FILES (>50KB):")
        for file_info in sorted(self.results["large_files"], key=lambda x: x["size"], reverse=True)[:10]:
            print(f"  • {file_info['file']}: {file_info['size']:,} bytes")
        
        # Optimization Opportunities
        print(f"\n💡 OPTIMIZATION OPPORTUNITIES:")
        for opp in self.results["optimization_opportunities"]:
            print(f"  • [{opp['priority'].upper()}] {opp['description']}")
            print(f"    Files affected: {opp['files_affected']}")
            print(f"    Estimated savings: {opp['estimated_savings']}")
            print()

if __name__ == "__main__":
    analyzer = PerformanceAnalyzer()
    results = analyzer.run_full_analysis()
    analyzer.print_report()
    
    # Save results to JSON
    with open("/opt/projects/knowledgehub/performance_analysis_results.json", "w") as f:
        # Convert defaultdict to regular dict for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, defaultdict):
                json_results[key] = dict(value)
            else:
                json_results[key] = value
        json.dump(json_results, f, indent=2, default=str)
    
    print(f"\n📁 Full results saved to: performance_analysis_results.json")