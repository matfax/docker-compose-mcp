"""Integration tests for MCP server functionality."""

import builtins
import contextlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

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
        with patch("docker_compose_mcp.docker.from_env"):
            manager = DockerComposeManager(test_project_dir)

            # Mock all methods
            manager.start_compose = MagicMock(return_value=True)
            manager.stop_compose = MagicMock(return_value=True)
            manager.restart_compose = MagicMock(return_value=True)
            manager.get_compose_logs = MagicMock(return_value="Test logs")
            manager.get_compose_status = MagicMock(
                return_value={
                    "project_name": "test-project",
                    "compose_file_exists": True,
                    "services": [{"Name": "test-service", "State": "running"}],
                    "containers": [
                        {
                            "name": "test-project_test-service_1",
                            "status": "running",
                            "health": "healthy",
                            "ports": {},
                        }
                    ],
                }
            )

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
        with patch("docker_compose_mcp.compose_manager", mock_compose_manager):
            result = await call_tool("get_compose_status", {})

            assert len(result) == 1
            assert result[0].type == "text"
            assert "test-project" in result[0].text
            assert "running" in result[0].text
            mock_compose_manager.get_compose_status.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_compose_logs_tool(self, mock_compose_manager):
        """Test get_compose_logs MCP tool."""
        with patch("docker_compose_mcp.compose_manager", mock_compose_manager):
            result = await call_tool("get_compose_logs", {"service": "test-service", "tail": 50})

            assert len(result) == 1
            assert result[0].type == "text"
            assert "Test logs" in result[0].text
            mock_compose_manager.get_compose_logs.assert_called_once_with("test-service", 50)

    @pytest.mark.asyncio
    async def test_get_compose_logs_tool_defaults(self, mock_compose_manager):
        """Test get_compose_logs MCP tool with default parameters."""
        with patch("docker_compose_mcp.compose_manager", mock_compose_manager):
            result = await call_tool("get_compose_logs", {})

            assert len(result) == 1
            assert result[0].type == "text"
            assert "Test logs" in result[0].text
            mock_compose_manager.get_compose_logs.assert_called_once_with(None, 100)

    @pytest.mark.asyncio
    async def test_restart_compose_tool(self, mock_compose_manager):
        """Test restart_compose MCP tool."""
        with patch("docker_compose_mcp.compose_manager", mock_compose_manager):
            result = await call_tool("restart_compose", {})

            assert len(result) == 1
            assert result[0].type == "text"
            assert "restarted successfully" in result[0].text
            mock_compose_manager.restart_compose.assert_called_once()

    @pytest.mark.asyncio
    async def test_restart_compose_tool_failure(self, mock_compose_manager):
        """Test restart_compose MCP tool when restart fails."""
        mock_compose_manager.restart_compose.return_value = False

        with patch("docker_compose_mcp.compose_manager", mock_compose_manager):
            result = await call_tool("restart_compose", {})

            assert len(result) == 1
            assert result[0].type == "text"
            assert "Failed to restart" in result[0].text

    @pytest.mark.asyncio
    async def test_unknown_tool(self, mock_compose_manager):
        """Test calling unknown MCP tool returns error message."""
        with patch("docker_compose_mcp.compose_manager", mock_compose_manager):
            result = await call_tool("unknown_tool", {})

            assert len(result) == 1
            assert result[0].type == "text"
            assert "Error executing tool 'unknown_tool'" in result[0].text
            assert "Unknown tool" in result[0].text


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

    @pytest.mark.skipif(not Path("/var/run/docker.sock").exists(), reason="Docker not available")
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
            with contextlib.suppress(builtins.BaseException):
                manager.stop_compose()
            raise e
