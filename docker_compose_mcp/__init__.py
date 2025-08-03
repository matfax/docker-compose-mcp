#!/usr/bin/env python3
"""
MCP Server for Docker Compose Lifecycle Management.

Manages Docker Compose startup, status monitoring, and graceful shutdown
with comprehensive type safety, security, and error handling.
"""

import argparse
import asyncio
import atexit
import json
import logging
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import docker
from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

# Logging will be configured in main() based on debug flag
logger = logging.getLogger(__name__)


class DockerComposeError(Exception):
    """Custom exception for Docker Compose operations."""


class DockerComposeManager:
    """Manages Docker Compose lifecycle and operations with security and type safety.

    This class provides a secure interface to Docker Compose operations,
    with proper input validation, error handling, and subprocess security.

    Attributes:
        project_dir: Resolved path to the Docker Compose project directory
        compose_file: Path to the Docker Compose file
        docker_client: Docker client instance for container operations
        project_name: Name of the Docker Compose project
    """

    def __init__(self, project_dir: str | Path, compose_file: str = "docker-compose.yml") -> None:
        """Initialize Docker Compose Manager.

        Args:
            project_dir: Path to the Docker Compose project directory
            compose_file: Name of the Docker Compose file

        Raises:
            FileNotFoundError: If the compose file doesn't exist
            DockerComposeError: If Docker client initialization fails
        """
        self.project_dir = Path(project_dir).resolve()
        self.compose_file = self.project_dir / compose_file
        self.project_name = self.project_dir.name

        # Validate project directory exists and is accessible
        if not self.project_dir.exists():
            raise FileNotFoundError(f"Project directory not found: {self.project_dir}")
        if not self.project_dir.is_dir():
            raise NotADirectoryError(f"Project path is not a directory: {self.project_dir}")

        # Validate compose file exists
        if not self.compose_file.exists():
            raise FileNotFoundError(f"Docker Compose file not found: {self.compose_file}")

        # Initialize Docker client with error handling
        try:
            self.docker_client = docker.from_env()
            # Test Docker connection
            self.docker_client.ping()
        except Exception as docker_error:
            raise DockerComposeError(f"Failed to initialize Docker client: {docker_error}") from docker_error

        # Load environment variables from project directory
        env_file = self.project_dir / "docker.env"
        if env_file.exists():
            try:
                load_dotenv(env_file)
                logger.info(f"Loaded environment from {env_file}")
            except Exception as env_error:
                logger.warning(f"Failed to load environment from {env_file}: {env_error}")

    def _run_compose_command(
        self, command: list[str], timeout: float = 60.0
    ) -> subprocess.CompletedProcess[str]:
        """Securely run a Docker Compose command with proper validation.

        Args:
            command: List of command arguments (will be validated)
            timeout: Command timeout in seconds

        Returns:
            CompletedProcess result

        Raises:
            DockerComposeError: If command execution fails
            ValueError: If command contains invalid arguments
        """
        # Validate command arguments to prevent injection
        if not command or not isinstance(command, list):
            raise ValueError("Command must be a non-empty list")

        # Ensure command starts with docker compose (v2 only)
        if len(command) < 2 or command[0] != "docker" or command[1] != "compose":
            raise ValueError("Command must start with 'docker compose'")

        # Validate all arguments are strings and don't contain shell metacharacters
        for arg in command:
            if not isinstance(arg, str):
                raise ValueError(f"All command arguments must be strings, got: {type(arg)}")
            # Basic validation - could be enhanced based on needs
            if any(char in arg for char in [";", "&", "|", "`", "$", "(", ")"]):
                raise ValueError(f"Command argument contains potentially unsafe characters: {arg}")

        try:
            logger.debug(f"Executing command: {' '.join(shlex.quote(arg) for arg in command)}")
            return subprocess.run(
                command,
                cwd=self.project_dir,
                capture_output=True,
                text=True,
                check=True,
                timeout=timeout,
                env=dict(os.environ),  # Use clean environment copy
            )
        except subprocess.TimeoutExpired as timeout_error:
            raise DockerComposeError(
                f"Command timed out after {timeout}s: {' '.join(command)}"
            ) from timeout_error
        except subprocess.CalledProcessError as called_process_error:
            raise DockerComposeError(
                f"Command failed: {called_process_error.stderr or called_process_error.stdout or 'Unknown error'}"
            ) from called_process_error
        except Exception as run_error:
            raise DockerComposeError(f"Unexpected error running command: {run_error}") from run_error

    def start_compose(self) -> bool:
        """Start Docker Compose services.

        Returns:
            True if services started successfully, False otherwise
        """
        try:
            logger.info("Starting Docker Compose services...")
            result = self._run_compose_command(["docker", "compose", "up", "-d"])
            logger.info(f"Docker Compose started successfully: {result.stdout.strip()}")
            return True
        except DockerComposeError as start_error:
            logger.error(f"Failed to start Docker Compose: {start_error}")
            return False

    def stop_compose(self) -> bool:
        """Stop Docker Compose services.

        Returns:
            True if services stopped successfully, False otherwise
        """
        try:
            logger.info("Stopping Docker Compose services...")
            result = self._run_compose_command(["docker", "compose", "down"])
            logger.info(f"Docker Compose stopped successfully: {result.stdout.strip()}")
            return True
        except DockerComposeError as stop_error:
            logger.error(f"Failed to stop Docker Compose: {stop_error}")
            return False

    def get_compose_status(self) -> dict[str, Any]:
        """Get status of Docker Compose services and containers.

        Returns:
            Dictionary containing service status, container information, and metadata
        """
        base_status = {
            "services": [],
            "containers": [],
            "compose_file_exists": self.compose_file.exists(),
            "project_name": self.project_name,
        }

        try:
            # Get compose services status
            result = self._run_compose_command(["docker", "compose", "ps", "--format", "json"])

            services_info: list[dict[str, Any]] = []
            if result.stdout.strip():
                # Parse each line as JSON (docker-compose ps outputs one JSON per line)
                for line in result.stdout.strip().split("\n"):
                    line = line.strip()
                    if line:
                        try:
                            services_info.append(json.loads(line))
                        except json.JSONDecodeError as json_error:
                            logger.warning(f"Failed to parse service JSON: {line} - {json_error}")

            base_status["services"] = services_info

        except DockerComposeError as status_error:
            logger.error(f"Failed to get compose status: {status_error}")
            base_status["error"] = str(status_error)

        # Get container health status
        try:
            containers: list[dict[str, Any]] = []
            for container in self.docker_client.containers.list(all=True):
                # More precise project matching
                container_labels = container.labels or {}
                if (
                    container_labels.get("com.docker.compose.project") == self.project_name
                    or self.project_name in container.name
                ):
                    health = container.attrs.get("State", {}).get("Health", {})
                    containers.append(
                        {
                            "name": container.name,
                            "status": container.status,
                            "health": health.get("Status", "none"),
                            "ports": container.ports or {},
                            "labels": container_labels,
                        }
                    )
            base_status["containers"] = containers

        except Exception as container_error:
            logger.warning(f"Could not get container details: {container_error}")

        return base_status

    def get_compose_logs(self, service: str | None = None, tail: int = 100) -> str:
        """Get logs from Docker Compose services.

        Args:
            service: Specific service name (optional)
            tail: Number of log lines to retrieve (default: 100, max: 10000)

        Returns:
            Log output as string
        """
        # Validate tail parameter
        if not isinstance(tail, int) or tail < 1 or tail > 10000:
            raise ValueError("Tail must be an integer between 1 and 10000")

        # Validate service name if provided
        if service is not None:
            if not isinstance(service, str) or not service.strip():
                raise ValueError("Service name must be a non-empty string")
            # Basic validation for service name (alphanumeric, hyphens, underscores)
            if not all(c.isalnum() or c in "-_" for c in service):
                raise ValueError("Service name contains invalid characters")

        try:
            cmd = ["docker", "compose", "logs", "--tail", str(tail)]
            if service:
                cmd.append(service.strip())

            result = self._run_compose_command(cmd)
            return result.stdout
        except DockerComposeError as logs_error:
            logger.error(f"Failed to get compose logs: {logs_error}")
            return f"Error getting logs: {logs_error}"

    def restart_compose(self) -> bool:
        """Restart Docker Compose services.

        Returns:
            True if services restarted successfully, False otherwise
        """
        try:
            logger.info("Restarting Docker Compose services...")
            result = self._run_compose_command(["docker", "compose", "restart"])
            logger.info(f"Docker Compose restarted successfully: {result.stdout.strip()}")
            return True
        except DockerComposeError as restart_error:
            logger.error(f"Failed to restart Docker Compose: {restart_error}")
            return False


# Global instance - will be initialized in main()
compose_manager: DockerComposeManager | None = None

# MCP Server setup
server = Server("docker-compose-lifecycle")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available MCP tools."""
    return [
        Tool(
            name="get_compose_status",
            description="Get the current status of Docker Compose services and containers",
            inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
        ),
        Tool(
            name="get_compose_logs",
            description="Get logs from Docker Compose services",
            inputSchema={
                "type": "object",
                "properties": {
                    "service": {
                        "type": "string",
                        "description": (
                            "Specific service name (optional, gets all services if not specified)"
                        ),
                    },
                    "tail": {
                        "type": "integer",
                        "description": "Number of log lines to retrieve (default: 100)",
                        "default": 100,
                    },
                },
                "additionalProperties": False,
            },
        ),
        Tool(
            name="restart_compose",
            description="Restart Docker Compose services",
            inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle MCP tool calls with proper error handling and validation.

    Args:
        name: Name of the tool to call
        arguments: Arguments for the tool

    Returns:
        List of TextContent responses

    Raises:
        ValueError: If tool name is unknown or arguments are invalid
        RuntimeError: If compose_manager is not initialized
    """
    if compose_manager is None:
        raise RuntimeError("Docker Compose Manager not initialized")

    try:
        if name == "get_compose_status":
            status = compose_manager.get_compose_status()
            return [
                TextContent(
                    type="text", text=f"Docker Compose Status:\n\n{format_status_output(status)}"
                )
            ]

        if name == "get_compose_logs":
            service = arguments.get("service")
            tail = arguments.get("tail", 100)

            # Validate arguments
            if not isinstance(tail, int):
                raise ValueError("Tail argument must be an integer")
            if service is not None and not isinstance(service, str):
                raise ValueError("Service argument must be a string")

            logs = compose_manager.get_compose_logs(service, tail)
            return [TextContent(type="text", text=f"Docker Compose Logs:\n\n{logs}")]

        if name == "restart_compose":
            success = compose_manager.restart_compose()
            message = (
                "Docker Compose services restarted successfully"
                if success
                else "Failed to restart Docker Compose services"
            )
            return [TextContent(type="text", text=message)]

        raise ValueError(f"Unknown tool: {name}")

    except Exception as tool_error:
        logger.error(f"Error executing tool '{name}': {tool_error}")
        return [TextContent(type="text", text=f"Error executing tool '{name}': {tool_error}")]


def format_status_output(status: dict[str, Any]) -> str:
    """Format status output for display with proper error handling.

    Args:
        status: Status dictionary from get_compose_status

    Returns:
        Formatted status string
    """
    output: list[str] = []

    output.append(f"Project: {status.get('project_name', 'unknown')}")
    output.append(f"Compose file exists: {status.get('compose_file_exists', False)}")

    if "error" in status:
        output.append(f"Error: {status['error']}")

    services = status.get("services", [])
    if services:
        output.append("\nServices:")
        for service in services:
            if isinstance(service, dict):
                name = service.get("Name", "unknown")
                state = service.get("State", "unknown")
                output.append(f"  - {name}: {state}")

    containers = status.get("containers", [])
    if containers:
        output.append("\nContainers:")
        for container in containers:
            if isinstance(container, dict):
                name = container.get("name", "unknown")
                container_status = container.get("status", "unknown")
                health = container.get("health", "none")
                output.append(f"  - {name}: {container_status} (health: {health})")

                ports = container.get("ports", {})
                if isinstance(ports, dict) and ports:
                    for port_info in ports.values():
                        if isinstance(port_info, list):
                            for port in port_info:
                                if isinstance(port, dict):
                                    host_ip = port.get("HostIp", "localhost")  # nosec B104
                                    host_port = port.get("HostPort", "N/A")
                                    output.append(f"    Port: {host_ip}:{host_port}")

    return "\n".join(output)


def startup_hook() -> None:
    """Hook to run on server startup.

    Raises:
        RuntimeError: If compose_manager is not initialized
    """
    if compose_manager is None:
        raise RuntimeError("Docker Compose Manager not initialized")

    logger.info("MCP Server starting up - initializing Docker Compose...")
    try:
        if not compose_manager.start_compose():
            logger.warning("Failed to start Docker Compose on startup")
        else:
            logger.info("Docker Compose started successfully on startup")
    except Exception as startup_hook_error:
        logger.error(f"Unexpected error during startup: {startup_hook_error}")
        raise


def shutdown_hook() -> None:
    """Hook to run on server shutdown."""
    if compose_manager is None:
        logger.warning("Docker Compose Manager not initialized, skipping shutdown")
        return

    logger.info("MCP Server shutting down - stopping Docker Compose...")
    try:
        if not compose_manager.stop_compose():
            logger.warning("Failed to stop Docker Compose on shutdown")
        else:
            logger.info("Docker Compose stopped successfully on shutdown")
    except Exception as shutdown_error:
        logger.error(f"Error during shutdown: {shutdown_error}")


def signal_handler(signum: int, frame: Any) -> None:
    """Handle shutdown signals gracefully.

    Args:
        signum: Signal number
        frame: Current stack frame
    """
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    try:
        shutdown_hook()
    except Exception as signal_error:
        logger.error(f"Error during signal handling: {signal_error}")
    finally:
        sys.exit(0)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments with validation.

    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="MCP Server that auto-starts Docker Compose services and provides monitoring tools",  # noqa: E501
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  %(prog)s /path/to/project
  %(prog)s /path/to/project --compose-file docker-compose.dev.yml
  %(prog)s /path/to/project --debug --log-level DEBUG""",
    )
    parser.add_argument("project_dir", help="Path to the Docker Compose project directory")
    parser.add_argument(
        "--compose-file",
        default="docker-compose.yml",
        help="Name of the Docker Compose file (default: docker-compose.yml)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set the logging level (default: INFO)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with stderr logging (use for troubleshooting)",
    )
    return parser.parse_args()


async def main() -> None:
    """Main server entry point with comprehensive error handling.

    Raises:
        SystemExit: On initialization or startup failures
    """
    global compose_manager

    # Parse command line arguments
    try:
        args = parse_args()
    except Exception as parse_error:
        logger.error(f"Failed to parse arguments: {parse_error}")
        sys.exit(1)

    # Configure logging based on debug flag and log level
    log_level = getattr(logging, args.log_level, logging.INFO)

    if args.debug:
        # Enable stderr logging for debug mode
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(sys.stderr)],
            force=True,  # Override any existing configuration
        )
        logger.info(f"Debug mode enabled - Log level set to {args.log_level}")
    else:
        # Use NullHandler to avoid stderr (MCP clients interpret stderr as errors)
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.NullHandler()],
            force=True,
        )

    logging.getLogger().setLevel(log_level)

    # Initialize compose manager
    try:
        compose_manager = DockerComposeManager(args.project_dir, args.compose_file)
        logger.info(
            f"Initialized Docker Compose manager for project: {compose_manager.project_name}"
        )
    except (FileNotFoundError, NotADirectoryError) as init_file_error:
        logger.error(f"Initialization failed: {init_file_error}")
        sys.exit(1)
    except DockerComposeError as init_compose_error:
        logger.error(f"Docker Compose initialization failed: {init_compose_error}")
        sys.exit(1)
    except Exception as init_error:
        logger.error(f"Unexpected error during initialization: {init_error}")
        sys.exit(1)

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Register shutdown hook
    atexit.register(shutdown_hook)

    # Run startup hook
    try:
        startup_hook()
    except Exception as startup_error:
        logger.error(f"Startup failed: {startup_error}")
        sys.exit(1)

    # Start MCP server
    logger.info("Starting MCP server...")
    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())
    except Exception as server_error:
        logger.error(f"MCP server error: {server_error}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
        sys.exit(0)
    except Exception as main_error:
        logger.error(f"Unexpected error: {main_error}")
        sys.exit(1)
