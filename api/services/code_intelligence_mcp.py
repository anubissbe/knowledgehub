#!/usr/bin/env python3
"""
Code Intelligence MCP Server - Exposes Serena-inspired tools to Claude Code
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

from mcp.server import Server, NotificationOptions, RequestOptions
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from code_intelligence_service import code_intelligence_service

logger = logging.getLogger(__name__)

# Initialize MCP server
app = Server("knowledgehub-code-intelligence")

# Store active project context
active_project = None


@app.list_tools()
async def list_tools() -> List[Tool]:
    """List all available code intelligence tools"""
    return [
        Tool(
            name="activate_project",
            description="Activate a project for code intelligence analysis",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_path": {
                        "type": "string",
                        "description": "Path to the project root directory"
                    }
                },
                "required": ["project_path"]
            }
        ),
        Tool(
            name="get_symbols_overview",
            description="Get an overview of code symbols (classes, functions, etc.) in a file or project",
            inputSchema={
                "type": "object",
                "properties": {
                    "relative_path": {
                        "type": "string",
                        "description": "Relative path to file (optional, omit for project overview)"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="find_symbol",
            description="Find a specific symbol by name or path (e.g., 'MyClass/my_method')",
            inputSchema={
                "type": "object",
                "properties": {
                    "name_path": {
                        "type": "string",
                        "description": "Name or hierarchical path of the symbol"
                    },
                    "include_body": {
                        "type": "boolean",
                        "description": "Include the symbol's source code",
                        "default": False
                    },
                    "include_references": {
                        "type": "boolean",
                        "description": "Include references to this symbol",
                        "default": False
                    }
                },
                "required": ["name_path"]
            }
        ),
        Tool(
            name="find_references",
            description="Find all references to a symbol in the project",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol_name": {
                        "type": "string",
                        "description": "Name of the symbol to find references for"
                    }
                },
                "required": ["symbol_name"]
            }
        ),
        Tool(
            name="replace_symbol",
            description="Replace a symbol's body with new code",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol_path": {
                        "type": "string",
                        "description": "Path to the symbol (e.g., 'MyClass/my_method')"
                    },
                    "new_body": {
                        "type": "string",
                        "description": "New code to replace the symbol body"
                    }
                },
                "required": ["symbol_path", "new_body"]
            }
        ),
        Tool(
            name="search_pattern",
            description="Search for a pattern in project files",
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Pattern to search for (regex supported)"
                    },
                    "file_pattern": {
                        "type": "string",
                        "description": "File pattern to search in (e.g., '*.py')"
                    },
                    "context_lines": {
                        "type": "integer",
                        "description": "Number of context lines to show",
                        "default": 2
                    }
                },
                "required": ["pattern"]
            }
        ),
        Tool(
            name="write_memory",
            description="Write project-specific memory for future reference",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the memory item"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to remember (markdown format)"
                    }
                },
                "required": ["name", "content"]
            }
        ),
        Tool(
            name="read_memory",
            description="Read a previously saved project memory",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the memory item to read"
                    }
                },
                "required": ["name"]
            }
        ),
        Tool(
            name="list_memories",
            description="List all saved memories for the current project",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="insert_after_symbol",
            description="Insert code after a symbol definition",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol_path": {
                        "type": "string",
                        "description": "Path to the symbol to insert after"
                    },
                    "code": {
                        "type": "string",
                        "description": "Code to insert"
                    }
                },
                "required": ["symbol_path", "code"]
            }
        ),
        Tool(
            name="insert_before_symbol",
            description="Insert code before a symbol definition",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol_path": {
                        "type": "string",
                        "description": "Path to the symbol to insert before"
                    },
                    "code": {
                        "type": "string",
                        "description": "Code to insert"
                    }
                },
                "required": ["symbol_path", "code"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Execute a code intelligence tool"""
    global active_project
    
    try:
        if name == "activate_project":
            project_path = arguments["project_path"]
            context = await code_intelligence_service.activate_project(project_path)
            active_project = project_path
            
            return [TextContent(
                type="text",
                text=f"Project activated: {context.project_root}\n"
                     f"Language: {context.language}\n"
                     f"Framework: {context.framework or 'None detected'}\n"
                     f"Dependencies: {len(context.dependencies)}\n"
                     f"Memories: {len(context.memory_items)}"
            )]
        
        # Ensure project is activated for other tools
        if not active_project:
            return [TextContent(
                type="text",
                text="Error: No project activated. Use 'activate_project' first."
            )]
        
        if name == "get_symbols_overview":
            file_path = arguments.get("relative_path")
            if file_path:
                file_path = str(Path(active_project) / file_path)
            
            symbols = await code_intelligence_service.get_symbols_overview(active_project, file_path)
            
            # Format symbols for display
            output = []
            for symbol in symbols:
                output.append(f"📦 {symbol['name']} ({symbol['kind']}) at {symbol['path']}:{symbol['line_start']}")
                if symbol.get('children'):
                    for child in symbol['children']:
                        output.append(f"  └─ {child['name']} ({child['kind']}) at line {child['line_start']}")
            
            return [TextContent(
                type="text",
                text="\n".join(output) if output else "No symbols found"
            )]
        
        elif name == "find_symbol":
            symbol = await code_intelligence_service.find_symbol(
                active_project,
                arguments["name_path"],
                arguments.get("include_body", False),
                arguments.get("include_references", False)
            )
            
            if symbol:
                output = [
                    f"Found: {symbol['name']} ({symbol['kind']})",
                    f"Path: {symbol['path']}:{symbol['line_start']}-{symbol['line_end']}",
                    f"Name Path: {symbol['name_path']}"
                ]
                
                if symbol.get('docstring'):
                    output.append(f"Docstring: {symbol['docstring']}")
                
                if symbol.get('body'):
                    output.append(f"\nCode:\n{symbol['body']}")
                
                if symbol.get('references'):
                    output.append(f"\nReferences ({len(symbol['references'])}):")
                    for ref in symbol['references'][:5]:
                        output.append(f"  - {ref['file']}:{ref['line']}: {ref['text'][:80]}")
                
                return [TextContent(type="text", text="\n".join(output))]
            else:
                return [TextContent(type="text", text=f"Symbol '{arguments['name_path']}' not found")]
        
        elif name == "find_references":
            refs = await code_intelligence_service._find_references(
                active_project,
                arguments["symbol_name"]
            )
            
            if refs:
                output = [f"Found {len(refs)} references to '{arguments['symbol_name']}':"]
                for ref in refs[:20]:  # Limit to 20 for readability
                    output.append(f"  {ref['file']}:{ref['line']}: {ref['text'][:80]}")
                
                if len(refs) > 20:
                    output.append(f"  ... and {len(refs) - 20} more")
                
                return [TextContent(type="text", text="\n".join(output))]
            else:
                return [TextContent(type="text", text=f"No references found for '{arguments['symbol_name']}'")]
        
        elif name == "replace_symbol":
            result = await code_intelligence_service.replace_symbol(
                active_project,
                arguments["symbol_path"],
                arguments["new_body"]
            )
            
            return [TextContent(type="text", text=result.get("message", "Operation completed"))]
        
        elif name == "search_pattern":
            results = await code_intelligence_service.search_pattern(
                active_project,
                arguments["pattern"],
                arguments.get("file_pattern"),
                arguments.get("context_lines", 2)
            )
            
            output = [f"Found pattern in {len(results)} files:"]
            for result in results[:10]:  # Limit files shown
                output.append(f"\n📄 {result['file']}:")
                for match in result['matches'][:5]:  # Limit matches per file
                    output.append(f"  Line {match['line']}: {match['text'][:100]}")
            
            if len(results) > 10:
                output.append(f"\n... and {len(results) - 10} more files")
            
            return [TextContent(type="text", text="\n".join(output))]
        
        elif name == "write_memory":
            result = await code_intelligence_service.save_memory(
                active_project,
                arguments["name"],
                arguments["content"]
            )
            
            return [TextContent(type="text", text=f"Memory '{arguments['name']}' saved successfully")]
        
        elif name == "read_memory":
            content = await code_intelligence_service.load_memory(
                active_project,
                arguments["name"]
            )
            
            if content:
                return [TextContent(type="text", text=content)]
            else:
                return [TextContent(type="text", text=f"Memory '{arguments['name']}' not found")]
        
        elif name == "list_memories":
            memories = await code_intelligence_service.list_memories(active_project)
            
            if memories:
                output = ["Project memories:"]
                for memory in memories:
                    output.append(f"  📝 {memory}")
                return [TextContent(type="text", text="\n".join(output))]
            else:
                return [TextContent(type="text", text="No memories saved for this project")]
        
        elif name in ["insert_after_symbol", "insert_before_symbol"]:
            # Find the symbol first
            symbol = await code_intelligence_service.find_symbol(
                active_project,
                arguments["symbol_path"],
                include_body=False
            )
            
            if not symbol:
                return [TextContent(type="text", text=f"Symbol '{arguments['symbol_path']}' not found")]
            
            # Read the file
            with open(symbol["path"], 'r') as f:
                lines = f.readlines()
            
            # Calculate insertion point
            if name == "insert_after_symbol":
                insert_line = symbol["line_end"]
            else:
                insert_line = symbol["line_start"] - 1
            
            # Get indentation
            if insert_line > 0 and insert_line < len(lines):
                ref_line = lines[insert_line - 1] if name == "insert_before_symbol" else lines[insert_line]
                indent = len(ref_line) - len(ref_line.lstrip())
            else:
                indent = 0
            
            # Prepare code with indentation
            code_lines = []
            for line in arguments["code"].split('\n'):
                if line.strip():
                    code_lines.append(' ' * indent + line + '\n')
                else:
                    code_lines.append('\n')
            
            # Insert the code
            lines[insert_line:insert_line] = code_lines
            
            # Write back
            with open(symbol["path"], 'w') as f:
                f.writelines(lines)
            
            position = "after" if name == "insert_after_symbol" else "before"
            return [TextContent(
                type="text",
                text=f"Code inserted {position} {arguments['symbol_path']} at line {insert_line + 1}"
            )]
        
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    
    except Exception as e:
        logger.error(f"Tool execution failed: {e}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def main():
    """Run the MCP server"""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())