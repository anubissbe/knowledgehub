#!/usr/bin/env python3
"""
Startup script for KnowledgeHub MCP Server.

This script provides an easy way to start the MCP server with proper
configuration and environment setup.

Usage:
    python start_server.py [--config config.json] [--log-level INFO]
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mcp_server.server import KnowledgeHubMCPServer


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Setup logging configuration."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured at level: {log_level}")
    
    return logger


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        return {}
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in config file: {e}")
        return {}


def validate_environment():
    """Validate that the environment is set up correctly."""
    issues = []
    
    # Check Python version
    if sys.version_info < (3, 8):
        issues.append("Python 3.8+ is required")
    
    # Check required packages
    required_packages = ['mcp', 'aiohttp', 'asyncio']
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            issues.append(f"Missing required package: {package}")
    
    # Check KnowledgeHub API accessibility
    try:
        # This is a basic check - in production you might want to ping the API
        api_path = project_root / "api"
        if not api_path.exists():
            issues.append("KnowledgeHub API directory not found")
    except Exception as e:
        issues.append(f"Environment check failed: {e}")
    
    return issues


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Start KnowledgeHub MCP Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python start_server.py                              # Start with default config
    python start_server.py --config custom.json        # Use custom config
    python start_server.py --log-level DEBUG           # Enable debug logging
    python start_server.py --validate-only             # Just validate environment
        """
    )
    
    parser.add_argument(
        '--config',
        default='config.json',
        help='Path to configuration file (default: config.json)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--log-file',
        help='Log to file in addition to console'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate environment and exit'
    )
    
    parser.add_argument(
        '--transport',
        choices=['stdio'],
        default='stdio',
        help='Transport type (default: stdio)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level, args.log_file)
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), args.config)
    config = load_config(config_path)
    
    if config:
        logger.info(f"Loaded configuration from: {config_path}")
    else:
        logger.warning("No configuration loaded, using defaults")
    
    # Validate environment
    logger.info("Validating environment...")
    issues = validate_environment()
    
    if issues:
        logger.error("Environment validation failed:")
        for issue in issues:
            logger.error(f"  - {issue}")
        
        if args.validate_only:
            sys.exit(1)
        else:
            logger.warning("Continuing despite validation issues...")
    else:
        logger.info("Environment validation passed")
        
        if args.validate_only:
            logger.info("Validation complete - environment is ready")
            sys.exit(0)
    
    # Create and start server
    logger.info("Initializing KnowledgeHub MCP Server...")
    
    try:
        server = KnowledgeHubMCPServer()
        
        logger.info(f"Starting server with {args.transport} transport...")
        logger.info("Server is ready to accept Claude Code connections")
        logger.info("To connect from Claude Code, add this server to your MCP configuration")
        
        # Start the server
        await server.run(transport_type=args.transport)
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal - shutting down server")
    except Exception as e:
        logger.error(f"Server error: {e}")
        logger.error("Check the logs for more details")
        sys.exit(1)
    finally:
        logger.info("Server shutdown complete")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Failed to start server: {e}")
        sys.exit(1)