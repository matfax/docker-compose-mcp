# Docker Compose MCP Server
[![Build and Test](https://github.com/matfax/docker-compose-mcp/actions/workflows/build.yml/badge.svg)](https://github.com/matfax/docker-compose-mcp/actions/workflows/build.yml)
![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fgithub.com%2Fmatfax%2Fdocker-compose-mcp%2Fraw%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)
![Docker](https://img.shields.io/badge/docker-compose-blue.svg)
[![ReviewDog](https://github.com/matfax/docker-compose-mcp/actions/workflows/reviewdog.yml/badge.svg?branch=main&event=push)](https://github.com/matfax/docker-compose-mcp/actions/workflows/reviewdog.yml?query=workflow%3Areviewdog+event%3Apush+branch%3Amain)
[![DeepSource](https://app.deepsource.com/gh/matfax/docker-compose-mcp.svg/?label=code+coverage&show_trend=true&token=8MocmlDi3f89SI7lmEmk1Kjc)](https://app.deepsource.com/gh/matfax/docker-compose-mcp/)
[![DeepSource](https://app.deepsource.com/gh/matfax/docker-compose-mcp.svg/?label=active+issues&show_trend=true&token=8MocmlDi3f89SI7lmEmk1Kjc)](https://app.deepsource.com/gh/matfax/docker-compose-mcp/)
[![DeepSource](https://app.deepsource.com/gh/matfax/docker-compose-mcp.svg/?label=resolved+issues&show_trend=true&token=8MocmlDi3f89SI7lmEmk1Kjc)](https://app.deepsource.com/gh/matfax/docker-compose-mcp/)
[![License](https://img.shields.io/badge/license-LGPL%203+-green.svg)](https://opensource.org/licenses/LGPL-3.0)

MCP server that provides lifecycle management for Docker Compose services with automatic startup, status monitoring, and graceful shutdown.

## Quick Start

```bash
# 1. Navigate to your Docker Compose project
cd /path/to/your/docker-compose-project

# 2. Run the MCP server (auto-starts compose services)
uv run --from git+https://github.com/matfax/docker-compose-mcp.git docker-compose-mcp .

# 3. Server provides MCP tools for status/logs/restart
# 4. Use Ctrl+C to stop (auto-runs docker-compose down)
```

## Installation

```bash
# Run directly from GitHub (recommended)
uv run --from git+https://github.com/matfax/docker-compose-mcp.git docker-compose-mcp /path/to/project

# Or install first
uv pip install git+https://github.com/matfax/docker-compose-mcp.git
docker-compose-mcp /path/to/project

# Development setup
git clone https://github.com/matfax/docker-compose-mcp.git
cd docker-compose-mcp
uv sync --dev
uv run docker-compose-mcp /path/to/project
```

**Requirements**: Python (see .python-version), Docker, Docker Compose, uv

## CLI Options

```bash
# Basic usage
docker-compose-mcp /path/to/project

# Use specific compose file
docker-compose-mcp /path/to/project --compose-file docker-compose.dev.yml

# Enable debug logging
docker-compose-mcp /path/to/project --debug --log-level DEBUG

# All options
docker-compose-mcp PROJECT_DIR [--compose-file FILE] [--log-level LEVEL] [--debug]
```

**Options:**
- `PROJECT_DIR`: Path to Docker Compose project directory
- `--compose-file`: Compose file name (default: docker-compose.yml)
- `--log-level`: Logging level - DEBUG, INFO, WARNING, ERROR (default: INFO)
- `--debug`: Enable debug mode with stderr logging

## Adding to Claude Code

Add to your MCP settings:

<details>
<summary>MCP Configuration (click to expand)</summary>

```json
{
  "mcpServers": {
    "docker-compose-mcp": {
      "command": "uv",
      "args": [
        "run",
        "--from",
        "git+https://github.com/matfax/docker-compose-mcp.git",
        "docker-compose-mcp",
        "/path/to/your/docker-compose-project"
      ]
    }
  }
}
```

For local installations:
```json
{
  "mcpServers": {
    "docker-compose-mcp": {
      "command": "uv",
      "args": [
        "run",
        "--from",
        "/path/to/docker-compose-mcp",
        "docker-compose-mcp",
        "/path/to/your/docker-compose-project"
      ]
    }
  }
}
```
</details>

## Available MCP Tools

### `get_compose_status`
Returns current status of all services and containers:
- Service states and health status
- Container information and port mappings
- Project details

### `get_compose_logs`
Retrieves Docker Compose logs with options:
- `service` (optional): Specific service name
- `tail` (optional): Number of log lines (default: 100)

### `restart_compose`
Restarts all Docker Compose services.

## Features

- **Automatic Lifecycle Management**: Auto-starts compose services on startup, stops on shutdown
- **Health Monitoring**: Tracks container health and port bindings  
- **Signal Handling**: Graceful shutdown with SIGINT/SIGTERM
- **Environment Loading**: Automatically loads `docker.env` from project directory
- **Security**: Input validation and subprocess security for all operations
- **Error Handling**: Comprehensive error handling with structured logging

## Environment Variables

Create a `docker.env` file in your project directory:
```env
POSTGRES_DB=myapp
POSTGRES_USER=user
POSTGRES_PASSWORD=password
```

The MCP server automatically loads these variables when starting Docker Compose.

## Lifecycle Management

- **On Startup**: Runs `docker-compose up -d` in the specified project directory
- **During Operation**: Provides MCP tools for monitoring and management
- **On Shutdown**: Gracefully runs `docker-compose down` to stop services
- **Signal Handling**: Responds to SIGINT/SIGTERM for clean shutdown

## Example Usage

```bash
# Basic usage with current directory
docker-compose-mcp .

# Use with specific project and compose file
docker-compose-mcp /home/user/my-app --compose-file docker-compose.prod.yml

# Debug mode for troubleshooting
docker-compose-mcp /home/user/my-app --debug --log-level DEBUG
```

The server will automatically start your Docker Compose services, provide MCP tools for management, and clean up when you exit.