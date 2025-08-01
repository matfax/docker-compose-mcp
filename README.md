# Docker Compose MCP Server

This MCP server provides lifecycle management for Docker Compose services with automatic startup, status monitoring, and graceful shutdown.

## Features

- **Automatic Startup**: Runs `docker-compose up -d` when the server starts
- **Status Monitoring**: Check service and container status via MCP tools
- **Log Retrieval**: Get Docker Compose logs through MCP interface
- **Graceful Shutdown**: Automatically runs `docker-compose down` on exit
- **Health Monitoring**: Tracks container health and port bindings

## Setup

1. **Install dependencies with uv**:
   ```bash
   uv venv
   uv pip install -r requirements.txt
   ```

2. **Run the MCP server**:
   ```bash
   uv run python mcp_docker_server.py
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

## Environment

The server loads environment variables from `docker.env` file for Docker Compose configuration.

## Usage Example

```bash
# Start the MCP server (auto-starts compose)
uv run python mcp_docker_server.py

# The server will:
# 1. Start Docker Compose services
# 2. Provide MCP tools for status/logs/restart
# 3. Stop services when you exit (Ctrl+C)
```