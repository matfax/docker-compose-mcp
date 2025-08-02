# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python MCP (Model Context Protocol) server that provides lifecycle management for Docker Compose services. The server automatically starts Docker Compose services on startup, provides MCP tools for monitoring and management, and gracefully shuts down services on exit.

## Development Commands

### Setup and Installation
```bash
# Development setup (recommended)
uv sync --dev

# Run from source
uv run docker-compose-mcp /path/to/docker-compose-project

# Install from repository 
uv pip install git+https://github.com/matfax/docker-compose-mcp.git
```

### Testing
```bash
# Run all tests
uv run pytest -v

# Run unit tests only
uv run pytest tests/test_docker_compose_manager.py -v

# Run MCP integration tests (without Docker)
uv run pytest tests/test_mcp_integration.py -v -k "not integration"

# Run Docker integration tests (requires Docker)
uv run pytest tests/test_mcp_integration.py::TestRealDockerIntegration -v -m integration

# Run tests with coverage
uv run pytest --cov=docker_compose_mcp --cov-report=term-missing
```

### Code Quality and Security
```bash
# Lint with Ruff
uv run ruff check .

# Format with Ruff
uv run ruff format .

# Check formatting
uv run ruff format --check .

# Type checking with MyPy
uv run mypy docker_compose_mcp/

# Security scanning with Bandit
uv run bandit -r docker_compose_mcp/

# Format with Black
uv run black docker_compose_mcp/ tests/

# Check Black formatting
uv run black --check --diff docker_compose_mcp/ tests/
```

## Architecture

### Core Components

- **`DockerComposeManager`** (`docker_compose_mcp/__init__.py:39-287`): Main class handling Docker Compose operations with security validation and error handling
- **MCP Server Setup** (`docker_compose_mcp/__init__.py:289-388`): MCP server with three tools for compose management
- **Signal Handling** (`docker_compose_mcp/__init__.py:439-488`): Graceful shutdown handling

### Key Features

- **Automatic Lifecycle Management**: Auto-starts compose services on server startup and stops them on shutdown
- **Security Validation**: Input validation and subprocess security for all Docker Compose commands
- **Comprehensive Error Handling**: Structured error handling with custom `DockerComposeError` exception
- **Environment Loading**: Automatically loads `docker.env` from project directory
- **Health Monitoring**: Tracks container health status and port bindings

### MCP Tools Available

1. **`get_compose_status`**: Returns status of services, containers, health, and port mappings
2. **`get_compose_logs`**: Retrieves logs with configurable tail and optional service filtering  
3. **`restart_compose`**: Restarts all Docker Compose services

### Project Structure
```
docker_compose_mcp/
├── __init__.py         # Main DockerComposeManager and MCP server code
├── __main__.py         # Module entry point
tests/
├── test_docker_compose_manager.py  # Unit tests for core functionality
├── test_mcp_integration.py        # MCP integration tests
└── fixtures/           # Test fixtures including sample docker-compose.yml
```

## Configuration

- **Python Version**: See `.python-version` file for required version
- **Dependencies**: Uses `uv` for dependency management, configured in `pyproject.toml`
- **Docker**: Requires Docker and Docker Compose to be installed and accessible
- **Environment**: Loads variables from `docker.env` file in target project directory

## Usage Pattern

The server is designed to be run with a path to any Docker Compose project:
```bash
uv run docker-compose-mcp /path/to/your/docker-compose-project
```

The server will automatically start the compose services and provide MCP tools for management until shutdown.