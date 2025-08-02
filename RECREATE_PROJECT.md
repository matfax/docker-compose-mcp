# How to Recreate the Docker Compose MCP Server Project

This document provides step-by-step instructions to recreate the entire Docker Compose MCP Server project from scratch.

## Project Overview

The Docker Compose MCP Server is a Model Context Protocol (MCP) server that provides lifecycle management for Docker Compose services with automatic startup, status monitoring, and graceful shutdown.

## Step-by-Step Recreation Instructions

### 1. Project Setup and Directory Structure

```bash
# Create project directory
mkdir docker-compose-mcp
cd docker-compose-mcp

# Initialize git repository
git init

# Create Python package structure
mkdir docker_compose_mcp
mkdir tests
mkdir tests/fixtures
mkdir tests/fixtures/test-project
mkdir .github
mkdir .github/workflows
```

### 2. Create Core Files

#### 2.1 Create `.gitignore`

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Environment files
*.env
.env.*
docker.env

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Logs
*.log
logs/

# Docker
.dockerignore

# MCP specific
*.mcprc
```

#### 2.2 Create `pyproject.toml`

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "docker-compose-mcp"
version = "0.1.0"
description = "MCP server for Docker Compose lifecycle management"
readme = "README.md"
license = "MIT"
requires-python = ">=3.10"
authors = [
    {name = "Matthias Fax", email = "mat@fax.fyi"},
]
keywords = ["mcp", "docker", "docker-compose", "server", "lifecycle"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Programming Language :: Python :: 3.14",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: System :: Systems Administration",
]

dependencies = [
    "mcp>=1.0.0",
    "docker>=7.0.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
    "pytest-mock>=3.14.0",
    "pytest-cov>=6.0.0",
    "mcp-use>=0.2.0",
]

[project.urls]
Homepage = "https://github.com/docker-compose-mcp"
Repository = "https://github.com/docker-compose-mcp.git"
Issues = "https://github.com/docker-compose-mcp/issues"

[project.scripts]
docker-compose-mcp = "docker_compose_mcp:main"

[tool.hatch.build.targets.wheel]
packages = ["docker_compose_mcp"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --cov=docker_compose_mcp --cov-report=term-missing"
asyncio_mode = "auto"

[tool.coverage.run]
source = ["docker_compose_mcp"]
omit = ["tests/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
]
```

#### 2.3 Create Main MCP Server (`docker_compose_mcp/__init__.py`)

This is the core MCP server implementation. Create the file with the following content:

```python
#!/usr/bin/env python3
"""
MCP Server for Docker Compose Lifecycle Management
Manages Docker Compose startup, status monitoring, and graceful shutdown.
"""

import argparse
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

class DockerComposeManager:
    """Manages Docker Compose lifecycle and operations."""
    
    def __init__(self, project_dir: Path, compose_file: str = "docker-compose.yml"):
        self.project_dir = Path(project_dir).resolve()
        self.compose_file = self.project_dir / compose_file
        self.docker_client = docker.from_env()
        self.project_name = self.project_dir.name
        
        # Load environment variables from project directory
        env_file = self.project_dir / 'docker.env'
        if env_file.exists():
            load_dotenv(env_file)
            logger.info(f"Loaded environment from {env_file}")
        
        # Validate compose file exists
        if not self.compose_file.exists():
            raise FileNotFoundError(f"Docker Compose file not found: {self.compose_file}")
        
    def start_compose(self) -> bool:
        """Start Docker Compose services."""
        try:
            logger.info("Starting Docker Compose services...")
            result = subprocess.run(
                ["docker-compose", "up", "-d"],
                cwd=self.project_dir,
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
                cwd=self.project_dir,
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
                cwd=self.project_dir,
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
                cwd=self.project_dir,
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
                cwd=self.project_dir,
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"Docker Compose restarted successfully: {result.stdout}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to restart Docker Compose: {e.stderr}")
            return False


# Global instance - will be initialized in main()
compose_manager = None

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

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MCP Server for Docker Compose lifecycle management"
    )
    parser.add_argument(
        "project_dir",
        help="Path to the Docker Compose project directory"
    )
    parser.add_argument(
        "--compose-file",
        default="docker-compose.yml",
        help="Name of the Docker Compose file (default: docker-compose.yml)"
    )
    return parser.parse_args()

async def main():
    """Main server entry point."""
    global compose_manager
    
    # Parse command line arguments
    args = parse_args()
    
    # Initialize compose manager
    try:
        compose_manager = DockerComposeManager(args.project_dir, args.compose_file)
        logger.info(f"Initialized Docker Compose manager for project: {compose_manager.project_name}")
    except FileNotFoundError as e:
        logger.error(f"Initialization failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during initialization: {e}")
        sys.exit(1)
    
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
```

#### 2.4 Create Module Entry Point (`docker_compose_mcp/__main__.py`)

```python
#!/usr/bin/env python3
"""
Entry point for running docker-compose-mcp as a module.
Usage: python -m docker_compose_mcp
"""

from . import main

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### 3. Create Test Infrastructure

#### 3.1 Create Test Package (`tests/__init__.py`)

```python
"""Tests for docker-compose-mcp package."""
```

#### 3.2 Create Test Fixtures (`tests/fixtures/test-project/docker-compose.yml`)

```yaml
services:
  hello-world:
    image: hello-world
    restart: "no"
  
  nginx:
    image: nginx:alpine
    ports:
      - "8080:80"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "wget", "--no-verbose", "--tries=1", "--spider", "http://localhost/"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

#### 3.3 Create Unit Tests (`tests/test_docker_compose_manager.py`)

```python
"""Unit tests for DockerComposeManager."""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from docker_compose_mcp import DockerComposeManager


class TestDockerComposeManager:
    """Test DockerComposeManager class."""
    
    @pytest.fixture
    def test_project_dir(self, tmp_path):
        """Create a temporary project directory with docker-compose.yml."""
        compose_content = """
services:
  test-service:
    image: hello-world
"""
        compose_file = tmp_path / "docker-compose.yml"
        compose_file.write_text(compose_content)
        return tmp_path
    
    @patch('docker_compose_mcp.docker.from_env')
    def test_init_with_valid_project(self, mock_docker, test_project_dir):
        """Test initialization with valid project directory."""
        mock_docker.return_value = MagicMock()
        
        manager = DockerComposeManager(test_project_dir)
        
        assert manager.project_dir == test_project_dir
        assert manager.compose_file == test_project_dir / "docker-compose.yml"
        assert manager.project_name == test_project_dir.name
        assert manager.compose_file.exists()
    
    @patch('docker_compose_mcp.docker.from_env')
    def test_init_missing_compose_file(self, mock_docker, tmp_path):
        """Test initialization fails with missing compose file."""
        mock_docker.return_value = MagicMock()
        
        with pytest.raises(FileNotFoundError, match="Docker Compose file not found"):
            DockerComposeManager(tmp_path)
    
    @patch('docker_compose_mcp.docker.from_env')
    @patch('docker_compose_mcp.load_dotenv')
    def test_init_loads_env_file(self, mock_load_dotenv, mock_docker, test_project_dir):
        """Test initialization loads docker.env if it exists."""
        mock_docker.return_value = MagicMock()
        
        # Create docker.env file
        env_file = test_project_dir / "docker.env"
        env_file.write_text("TEST_VAR=test_value\n")
        
        manager = DockerComposeManager(test_project_dir)
        
        mock_load_dotenv.assert_called_once_with(env_file)
    
    @patch('docker_compose_mcp.docker.from_env')
    @patch('docker_compose_mcp.subprocess.run')
    def test_start_compose_success(self, mock_run, mock_docker, test_project_dir):
        """Test successful Docker Compose start."""
        mock_docker.return_value = MagicMock()
        mock_run.return_value = MagicMock(stdout="Services started", stderr="")
        
        manager = DockerComposeManager(test_project_dir)
        result = manager.start_compose()
        
        assert result is True
        mock_run.assert_called_once_with(
            ["docker-compose", "up", "-d"],
            cwd=test_project_dir,
            capture_output=True,
            text=True,
            check=True
        )
    
    @patch('docker_compose_mcp.docker.from_env')
    @patch('docker_compose_mcp.subprocess.run')
    def test_start_compose_failure(self, mock_run, mock_docker, test_project_dir):
        """Test Docker Compose start failure."""
        mock_docker.return_value = MagicMock()
        mock_run.side_effect = subprocess.CalledProcessError(1, "docker-compose", stderr="Error")
        
        manager = DockerComposeManager(test_project_dir)
        result = manager.start_compose()
        
        assert result is False
    
    @patch('docker_compose_mcp.docker.from_env')
    @patch('docker_compose_mcp.subprocess.run')
    def test_stop_compose_success(self, mock_run, mock_docker, test_project_dir):
        """Test successful Docker Compose stop."""
        mock_docker.return_value = MagicMock()
        mock_run.return_value = MagicMock(stdout="Services stopped", stderr="")
        
        manager = DockerComposeManager(test_project_dir)
        result = manager.stop_compose()
        
        assert result is True
        mock_run.assert_called_once_with(
            ["docker-compose", "down"],
            cwd=test_project_dir,
            capture_output=True,
            text=True,
            check=True
        )
    
    @patch('docker_compose_mcp.docker.from_env')
    @patch('docker_compose_mcp.subprocess.run')
    def test_get_compose_status_success(self, mock_run, mock_docker, test_project_dir):
        """Test successful compose status retrieval."""
        mock_docker_client = MagicMock()
        mock_docker.return_value = mock_docker_client
        
        # Mock docker-compose ps output
        service_data = {"Name": "test-service", "State": "running"}
        mock_run.return_value = MagicMock(
            stdout=json.dumps(service_data) + "\n",
            stderr=""
        )
        
        # Mock container list
        mock_container = MagicMock()
        mock_container.name = f"{test_project_dir.name}_test-service_1"
        mock_container.status = "running"
        mock_container.attrs = {"State": {"Health": {"Status": "healthy"}}}
        mock_container.ports = {"80/tcp": [{"HostIp": "0.0.0.0", "HostPort": "8080"}]}
        mock_docker_client.containers.list.return_value = [mock_container]
        
        manager = DockerComposeManager(test_project_dir)
        status = manager.get_compose_status()
        
        assert status["project_name"] == test_project_dir.name
        assert status["compose_file_exists"] is True
        assert len(status["services"]) == 1
        assert status["services"][0]["Name"] == "test-service"
        assert len(status["containers"]) == 1
    
    @patch('docker_compose_mcp.docker.from_env')
    @patch('docker_compose_mcp.subprocess.run')
    def test_get_compose_logs_success(self, mock_run, mock_docker, test_project_dir):
        """Test successful compose logs retrieval."""
        mock_docker.return_value = MagicMock()
        mock_run.return_value = MagicMock(stdout="Service logs", stderr="")
        
        manager = DockerComposeManager(test_project_dir)
        logs = manager.get_compose_logs()
        
        assert logs == "Service logs"
        mock_run.assert_called_once_with(
            ["docker-compose", "logs", "--tail", "100"],
            cwd=test_project_dir,
            capture_output=True,
            text=True,
            check=True
        )
    
    @patch('docker_compose_mcp.docker.from_env')
    @patch('docker_compose_mcp.subprocess.run')
    def test_get_compose_logs_with_service(self, mock_run, mock_docker, test_project_dir):
        """Test compose logs retrieval for specific service."""
        mock_docker.return_value = MagicMock()
        mock_run.return_value = MagicMock(stdout="Service logs", stderr="")
        
        manager = DockerComposeManager(test_project_dir)
        logs = manager.get_compose_logs("test-service", 50)
        
        assert logs == "Service logs"
        mock_run.assert_called_once_with(
            ["docker-compose", "logs", "--tail", "50", "test-service"],
            cwd=test_project_dir,
            capture_output=True,
            text=True,
            check=True
        )
    
    @patch('docker_compose_mcp.docker.from_env')
    @patch('docker_compose_mcp.subprocess.run')
    def test_restart_compose_success(self, mock_run, mock_docker, test_project_dir):
        """Test successful Docker Compose restart."""
        mock_docker.return_value = MagicMock()
        mock_run.return_value = MagicMock(stdout="Services restarted", stderr="")
        
        manager = DockerComposeManager(test_project_dir)
        result = manager.restart_compose()
        
        assert result is True
        mock_run.assert_called_once_with(
            ["docker-compose", "restart"],
            cwd=test_project_dir,
            capture_output=True,
            text=True,
            check=True
        )
```

#### 3.4 Create MCP Integration Tests (`tests/test_mcp_integration.py`)

```python
"""Integration tests for MCP server functionality."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp_use import create_session

from docker_compose_mcp import DockerComposeManager, call_tool, list_tools


class TestMCPIntegration:
    """Test MCP server integration."""
    
    @pytest.fixture
    def test_project_dir(self, tmp_path):
        """Create a temporary project directory with docker-compose.yml."""
        compose_content = """
services:
  test-service:
    image: hello-world
  nginx:
    image: nginx:alpine
    ports:
      - "8080:80"
"""
        compose_file = tmp_path / "docker-compose.yml"
        compose_file.write_text(compose_content)
        return tmp_path
    
    @pytest.fixture
    def mock_compose_manager(self, test_project_dir):
        """Create a mocked DockerComposeManager."""
        with patch('docker_compose_mcp.docker.from_env'):
            manager = DockerComposeManager(test_project_dir)
            
            # Mock all methods
            manager.start_compose = MagicMock(return_value=True)
            manager.stop_compose = MagicMock(return_value=True)
            manager.restart_compose = MagicMock(return_value=True)
            manager.get_compose_logs = MagicMock(return_value="Test logs")
            manager.get_compose_status = MagicMock(return_value={
                "project_name": "test-project",
                "compose_file_exists": True,
                "services": [{"Name": "test-service", "State": "running"}],
                "containers": [{
                    "name": "test-project_test-service_1",
                    "status": "running",
                    "health": "healthy",
                    "ports": {}
                }]
            })
            
            return manager
    
    @pytest.mark.asyncio
    async def test_list_tools(self):
        """Test MCP tool listing."""
        tools = await list_tools()
        
        assert len(tools) == 3
        tool_names = [tool.name for tool in tools]
        assert "get_compose_status" in tool_names
        assert "get_compose_logs" in tool_names
        assert "restart_compose" in tool_names
    
    @pytest.mark.asyncio
    async def test_get_compose_status_tool(self, mock_compose_manager):
        """Test get_compose_status MCP tool."""
        # Patch the global compose_manager
        with patch('docker_compose_mcp.compose_manager', mock_compose_manager):
            result = await call_tool("get_compose_status", {})
            
            assert len(result) == 1
            assert result[0].type == "text"
            assert "test-project" in result[0].text
            assert "running" in result[0].text
            mock_compose_manager.get_compose_status.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_compose_logs_tool(self, mock_compose_manager):
        """Test get_compose_logs MCP tool."""
        with patch('docker_compose_mcp.compose_manager', mock_compose_manager):
            result = await call_tool("get_compose_logs", {"service": "test-service", "tail": 50})
            
            assert len(result) == 1
            assert result[0].type == "text"
            assert "Test logs" in result[0].text
            mock_compose_manager.get_compose_logs.assert_called_once_with("test-service", 50)
    
    @pytest.mark.asyncio
    async def test_get_compose_logs_tool_defaults(self, mock_compose_manager):
        """Test get_compose_logs MCP tool with default parameters."""
        with patch('docker_compose_mcp.compose_manager', mock_compose_manager):
            result = await call_tool("get_compose_logs", {})
            
            assert len(result) == 1
            assert result[0].type == "text"
            assert "Test logs" in result[0].text
            mock_compose_manager.get_compose_logs.assert_called_once_with(None, 100)
    
    @pytest.mark.asyncio
    async def test_restart_compose_tool(self, mock_compose_manager):
        """Test restart_compose MCP tool."""
        with patch('docker_compose_mcp.compose_manager', mock_compose_manager):
            result = await call_tool("restart_compose", {})
            
            assert len(result) == 1
            assert result[0].type == "text"
            assert "restarted successfully" in result[0].text
            mock_compose_manager.restart_compose.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_restart_compose_tool_failure(self, mock_compose_manager):
        """Test restart_compose MCP tool when restart fails."""
        mock_compose_manager.restart_compose.return_value = False
        
        with patch('docker_compose_mcp.compose_manager', mock_compose_manager):
            result = await call_tool("restart_compose", {})
            
            assert len(result) == 1
            assert result[0].type == "text"
            assert "Failed to restart" in result[0].text
    
    @pytest.mark.asyncio
    async def test_unknown_tool(self, mock_compose_manager):
        """Test calling unknown MCP tool raises ValueError."""
        with patch('docker_compose_mcp.compose_manager', mock_compose_manager):
            with pytest.raises(ValueError, match="Unknown tool"):
                await call_tool("unknown_tool", {})


@pytest.mark.integration
class TestRealDockerIntegration:
    """Integration tests with real Docker (requires Docker to be running)."""
    
    @pytest.fixture
    def test_project_dir(self, tmp_path):
        """Create a test project with a simple docker-compose.yml."""
        compose_content = """
services:
  hello-world:
    image: hello-world
    restart: "no"
"""
        compose_file = tmp_path / "docker-compose.yml"
        compose_file.write_text(compose_content)
        return tmp_path
    
    @pytest.mark.skipif(
        not Path("/var/run/docker.sock").exists(),
        reason="Docker not available"
    )
    def test_real_docker_compose_lifecycle(self, test_project_dir):
        """Test real Docker Compose lifecycle management."""
        try:
            # Initialize manager
            manager = DockerComposeManager(test_project_dir)
            
            # Test start
            start_result = manager.start_compose()
            assert start_result is True, "Failed to start Docker Compose"
            
            # Test status
            status = manager.get_compose_status()
            assert status["project_name"] == test_project_dir.name
            assert status["compose_file_exists"] is True
            
            # Test logs
            logs = manager.get_compose_logs()
            assert isinstance(logs, str)
            
            # Test stop
            stop_result = manager.stop_compose()
            assert stop_result is True, "Failed to stop Docker Compose"
            
        except Exception as e:
            # Clean up on any failure
            try:
                manager.stop_compose()
            except:
                pass
            raise e
```

### 4. Create GitHub Actions Workflow (`.github/workflows/test.yml`)

```yaml
name: Tests

on:
  push:
    branches: [ master, main ]
  pull_request:
    branches: [ master, main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12", "3.13", "3.14"]
      fail-fast: false

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}
        allow-prereleases: true

    - name: Install uv
      uses: astral-sh/setup-uv@v4
      with:
        version: "latest"

    - name: Install dependencies
      run: |
        uv sync --dev

    - name: Run unit tests
      run: |
        uv run pytest tests/test_docker_compose_manager.py -v

    - name: Run MCP integration tests
      run: |
        uv run pytest tests/test_mcp_integration.py -v -k "not integration"

    - name: Run real Docker integration tests
      run: |
        uv run pytest tests/test_mcp_integration.py::TestRealDockerIntegration -v -m integration
      continue-on-error: true  # Docker may not be available in all environments

  lint:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: "3.12"

    - name: Install uv
      uses: astral-sh/setup-uv@v4
      with:
        version: "latest"

    - name: Install dependencies
      run: |
        uv sync --dev

    - name: Check package can be imported
      run: |
        uv run python -c "import docker_compose_mcp; print('Package imports successfully')"
```

### 5. Create Renovate Configuration (`.github/renovate.json`)

```json
{
  "$schema": "https://docs.renovatebot.com/renovate-schema.json",
  "extends": [
    "config:base",
    ":dependencyDashboard",
    ":semanticCommits",
    ":automergeMinor"
  ],
  "python": {
    "enabled": true
  },
  "packageRules": [
    {
      "matchPackageNames": ["mcp", "docker", "python-dotenv"],
      "matchUpdateTypes": ["minor", "patch"],
      "automerge": true
    }
  ],
  "schedule": ["before 6am on Monday"],
  "timezone": "UTC"
}
```

### 6. Create README.md

```markdown
# Docker Compose MCP Server

This MCP server provides lifecycle management for Docker Compose services with automatic startup, status monitoring, and graceful shutdown.

## Features

- **Automatic Startup**: Runs `docker-compose up -d` when the server starts
- **Status Monitoring**: Check service and container status via MCP tools
- **Log Retrieval**: Get Docker Compose logs through MCP interface
- **Graceful Shutdown**: Automatically runs `docker-compose down` on exit
- **Health Monitoring**: Tracks container health and port bindings

## Installation

### Option 1: Install from GitHub (Recommended)
```bash
uv pip install git+https://github.com/docker-compose-mcp.git
```

### Option 2: Run directly without installation
```bash
uv run --from git+https://github.com/docker-compose-mcp.git docker-compose-mcp /path/to/your/project
```

### Option 3: Development setup
```bash
git clone https://github.com/docker-compose-mcp.git
cd docker-compose-mcp
uv venv
uv pip install -e .
```

## Usage

### Run the MCP server
```bash
# If installed
docker-compose-mcp /path/to/your/docker-compose-project

# Or as Python module
python -m docker_compose_mcp /path/to/your/docker-compose-project

# Or with uv run
uv run --from git+https://github.com/docker-compose-mcp.git docker-compose-mcp /path/to/your/project
```

## Available MCP Tools

### `get_compose_status`
Returns current status of all Docker Compose services and containers including:
- Service states
- Container health status
- Port mappings
- Project information

### `get_compose_logs`
Retrieves logs from Docker Compose services.
- `service` (optional): Specific service name
- `tail` (optional): Number of log lines (default: 100)

### `restart_compose`
Restarts all Docker Compose services.

## Lifecycle Management

- **On Startup**: Automatically runs `docker-compose up -d`
- **On Shutdown**: Gracefully runs `docker-compose down`
- **Signal Handling**: Responds to SIGINT/SIGTERM for clean shutdown

## Requirements

- Python 3.10+
- Docker and Docker Compose installed
- A `docker-compose.yml` file in your project directory

## Usage Example

```bash
# Navigate to your Docker Compose project
cd /path/to/your/docker-compose-project

# Run the MCP server (auto-starts compose)
docker-compose-mcp .

# The server will:
# 1. Start Docker Compose services
# 2. Provide MCP tools for status/logs/restart
# 3. Stop services when you exit (Ctrl+C)
```

## Environment

The server loads environment variables from `docker.env` file in the project directory for Docker Compose configuration.
```

### 7. Initialize the Project with uv

```bash
# Create virtual environment
uv venv

# Add dependencies properly
uv add mcp docker python-dotenv

# Add development dependencies
uv add --dev pytest pytest-asyncio pytest-mock pytest-cov mcp-use

# Lock dependencies
uv lock --upgrade
```

### 8. Git Operations

```bash
# Add all files
git add .

# First commit
git commit -m "Initial commit: Generic Docker Compose MCP Server

Add MCP server for Docker Compose lifecycle management with:
- Automatic compose up/down on server start/stop
- Status monitoring and log retrieval tools
- Graceful shutdown handling
- Python virtual environment setup with uv
- Works with any docker-compose.yml file

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# Add project directory argument feature
git commit -m "Remove docker.env dependency and add project directory argument

- Add command line argument parsing for project directory
- Update DockerComposeManager to work with specified project directory
- Load docker.env from project directory if it exists
- Validate compose file exists before starting
- Update all subprocess calls to use project directory as cwd

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# Add testing infrastructure
git commit -m "Add comprehensive testing infrastructure

- Update pyproject.toml with test dependencies and Python 3.13/3.14 support
- Create test fixtures with Docker Compose hello-world and nginx services
- Add unit tests with subprocess mocking for DockerComposeManager
- Add MCP integration tests using mcp-use library
- Create GitHub Actions workflow with Python 3.10-3.14 matrix testing
- Add Renovate configuration for automated dependency management
- Include both isolated unit tests and real Docker integration tests

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

### 9. Add MIT License

Create a `LICENSE` file with MIT license content (replace with actual MIT license text).

## Key Features Implemented

1. **MCP Server**: Full Model Context Protocol server implementation
2. **Docker Compose Management**: Lifecycle management with startup/shutdown hooks
3. **Command Line Interface**: Project directory argument support
4. **Testing Suite**: Comprehensive unit and integration tests
5. **CI/CD**: GitHub Actions workflow with Python 3.10-3.14 support
6. **Dependency Management**: Modern uv-based package management
7. **Auto-updates**: Renovate configuration for dependency updates

## Repository Structure

```
docker-compose-mcp/
├── .github/
│   ├── renovate.json
│   └── workflows/
│       └── test.yml
├── docker_compose_mcp/
│   ├── __init__.py
│   └── __main__.py
├── tests/
│   ├── __init__.py
│   ├── fixtures/
│   │   └── test-project/
│   │       └── docker-compose.yml
│   ├── test_docker_compose_manager.py
│   └── test_mcp_integration.py
├── .gitignore
├── LICENSE
├── pyproject.toml
├── README.md
├── uv.lock
└── RECREATE_PROJECT.md
```

This comprehensive guide allows you to recreate the entire Docker Compose MCP Server project from scratch, including all the features, testing infrastructure, and CI/CD setup that was developed.