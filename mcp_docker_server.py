#!/usr/bin/env python3
"""
MCP Server for Docker Compose Lifecycle Management
Manages Docker Compose startup, status monitoring, and graceful shutdown.
"""

import asyncio
import atexit
import logging
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import docker
from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from pydantic import BaseModel


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv('docker.env')

class DockerComposeManager:
    """Manages Docker Compose lifecycle and operations."""
    
    def __init__(self, compose_file: str = "docker-compose.yml"):
        self.compose_file = Path(compose_file)
        self.docker_client = docker.from_env()
        self.project_name = Path.cwd().name
        
    def start_compose(self) -> bool:
        """Start Docker Compose services."""
        try:
            logger.info("Starting Docker Compose services...")
            result = subprocess.run(
                ["docker-compose", "up", "-d"],
                cwd=Path.cwd(),
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"Docker Compose started successfully: {result.stdout}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start Docker Compose: {e.stderr}")
            return False
    
    def stop_compose(self) -> bool:
        """Stop Docker Compose services."""
        try:
            logger.info("Stopping Docker Compose services...")
            result = subprocess.run(
                ["docker-compose", "down"],
                cwd=Path.cwd(),
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"Docker Compose stopped successfully: {result.stdout}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to stop Docker Compose: {e.stderr}")
            return False
    
    def get_compose_status(self) -> Dict[str, Any]:
        """Get status of Docker Compose services."""
        try:
            # Get compose services status
            result = subprocess.run(
                ["docker-compose", "ps", "--format", "json"],
                cwd=Path.cwd(),
                capture_output=True,
                text=True,
                check=True
            )
            
            services_info = []
            if result.stdout.strip():
                import json
                # Parse each line as JSON (docker-compose ps outputs one JSON per line)
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        services_info.append(json.loads(line))
            
            # Get container health status
            containers = []
            try:
                for container in self.docker_client.containers.list(all=True):
                    if f"{self.project_name}" in container.name or "copilot-proxy" in container.name:
                        health = container.attrs.get('State', {}).get('Health', {})
                        containers.append({
                            'name': container.name,
                            'status': container.status,
                            'health': health.get('Status', 'none'),
                            'ports': container.ports
                        })
            except Exception as e:
                logger.warning(f"Could not get container details: {e}")
            
            return {
                'services': services_info,
                'containers': containers,
                'compose_file_exists': self.compose_file.exists(),
                'project_name': self.project_name
            }
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get compose status: {e.stderr}")
            return {
                'error': str(e),
                'services': [],
                'containers': [],
                'compose_file_exists': self.compose_file.exists(),
                'project_name': self.project_name
            }
    
    def get_compose_logs(self, service: Optional[str] = None, tail: int = 100) -> str:
        """Get logs from Docker Compose services."""
        try:
            cmd = ["docker-compose", "logs", "--tail", str(tail)]
            if service:
                cmd.append(service)
            
            result = subprocess.run(
                cmd,
                cwd=Path.cwd(),
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get compose logs: {e.stderr}")
            return f"Error getting logs: {e.stderr}"
    
    def restart_compose(self) -> bool:
        """Restart Docker Compose services."""
        try:
            logger.info("Restarting Docker Compose services...")
            result = subprocess.run(
                ["docker-compose", "restart"],
                cwd=Path.cwd(),
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"Docker Compose restarted successfully: {result.stdout}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to restart Docker Compose: {e.stderr}")
            return False


# Global instance
compose_manager = DockerComposeManager()

# MCP Server setup
server = Server("docker-compose-lifecycle")

@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available MCP tools."""
    return [
        Tool(
            name="get_compose_status",
            description="Get the current status of Docker Compose services and containers",
            inputSchema={
                "type": "object",
                "properties": {},
                "additionalProperties": False
            }
        ),
        Tool(
            name="get_compose_logs",
            description="Get logs from Docker Compose services",
            inputSchema={
                "type": "object",
                "properties": {
                    "service": {
                        "type": "string",
                        "description": "Specific service name (optional, gets all services if not specified)"
                    },
                    "tail": {
                        "type": "integer",
                        "description": "Number of log lines to retrieve (default: 100)",
                        "default": 100
                    }
                },
                "additionalProperties": False
            }
        ),
        Tool(
            name="restart_compose",
            description="Restart Docker Compose services",
            inputSchema={
                "type": "object",
                "properties": {},
                "additionalProperties": False
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle MCP tool calls."""
    
    if name == "get_compose_status":
        status = compose_manager.get_compose_status()
        return [TextContent(
            type="text",
            text=f"Docker Compose Status:\n\n{format_status_output(status)}"
        )]
    
    elif name == "get_compose_logs":
        service = arguments.get("service")
        tail = arguments.get("tail", 100)
        logs = compose_manager.get_compose_logs(service, tail)
        return [TextContent(
            type="text",
            text=f"Docker Compose Logs:\n\n{logs}"
        )]
    
    elif name == "restart_compose":
        success = compose_manager.restart_compose()
        message = "Docker Compose services restarted successfully" if success else "Failed to restart Docker Compose services"
        return [TextContent(
            type="text",
            text=message
        )]
    
    else:
        raise ValueError(f"Unknown tool: {name}")

def format_status_output(status: Dict[str, Any]) -> str:
    """Format status output for display."""
    output = []
    
    output.append(f"Project: {status['project_name']}")
    output.append(f"Compose file exists: {status['compose_file_exists']}")
    
    if 'error' in status:
        output.append(f"Error: {status['error']}")
    
    if status['services']:
        output.append("\nServices:")
        for service in status['services']:
            output.append(f"  - {service.get('Name', 'unknown')}: {service.get('State', 'unknown')}")
    
    if status['containers']:
        output.append("\nContainers:")
        for container in status['containers']:
            output.append(f"  - {container['name']}: {container['status']} (health: {container['health']})")
            if container['ports']:
                for port_info in container['ports'].values():
                    if port_info:
                        for port in port_info:
                            output.append(f"    Port: {port.get('HostIp', '0.0.0.0')}:{port.get('HostPort', 'N/A')}")
    
    return "\n".join(output)

def startup_hook():
    """Hook to run on server startup."""
    logger.info("MCP Server starting up - initializing Docker Compose...")
    if not compose_manager.start_compose():
        logger.warning("Failed to start Docker Compose on startup")
    else:
        logger.info("Docker Compose started successfully on startup")

def shutdown_hook():
    """Hook to run on server shutdown."""
    logger.info("MCP Server shutting down - stopping Docker Compose...")
    if not compose_manager.stop_compose():
        logger.warning("Failed to stop Docker Compose on shutdown")
    else:
        logger.info("Docker Compose stopped successfully on shutdown")

def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    shutdown_hook()
    sys.exit(0)

async def main():
    """Main server entry point."""
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Register shutdown hook
    atexit.register(shutdown_hook)
    
    # Run startup hook
    startup_hook()
    
    # Start MCP server
    logger.info("Starting MCP server...")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())