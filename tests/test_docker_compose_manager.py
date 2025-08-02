"""Unit tests for DockerComposeManager."""

import json
import subprocess
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

    @patch("docker_compose_mcp.docker.from_env")
    def test_init_with_valid_project(self, mock_docker, test_project_dir):
        """Test initialization with valid project directory."""
        mock_docker.return_value = MagicMock()

        manager = DockerComposeManager(test_project_dir)

        assert manager.project_dir == test_project_dir
        assert manager.compose_file == test_project_dir / "docker-compose.yml"
        assert manager.project_name == test_project_dir.name
        assert manager.compose_file.exists()

    @patch("docker_compose_mcp.docker.from_env")
    def test_init_missing_compose_file(self, mock_docker, tmp_path):
        """Test initialization fails with missing compose file."""
        mock_docker.return_value = MagicMock()

        with pytest.raises(FileNotFoundError, match="Docker Compose file not found"):
            DockerComposeManager(tmp_path)

    @patch("docker_compose_mcp.docker.from_env")
    @patch("docker_compose_mcp.load_dotenv")
    def test_init_loads_env_file(self, mock_load_dotenv, mock_docker, test_project_dir):
        """Test initialization loads docker.env if it exists."""
        mock_docker.return_value = MagicMock()

        # Create docker.env file
        env_file = test_project_dir / "docker.env"
        env_file.write_text("TEST_VAR=test_value\n")

        DockerComposeManager(test_project_dir)

        mock_load_dotenv.assert_called_once_with(env_file)

    @patch("docker_compose_mcp.docker.from_env")
    @patch("docker_compose_mcp.subprocess.run")
    def test_start_compose_success(self, mock_run, mock_docker, test_project_dir):
        """Test successful Docker Compose start."""
        mock_docker.return_value = MagicMock()
        mock_run.return_value = MagicMock(stdout="Services started", stderr="")

        manager = DockerComposeManager(test_project_dir)
        result = manager.start_compose()

        assert result is True
        # Verify the command was called with the right arguments
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args[0] == (["docker", "compose", "up", "-d"],)
        assert call_args[1]["cwd"] == test_project_dir
        assert call_args[1]["capture_output"] is True
        assert call_args[1]["text"] is True
        assert call_args[1]["check"] is True
        assert call_args[1]["timeout"] == 60.0
        assert "env" in call_args[1]

    @patch("docker_compose_mcp.docker.from_env")
    @patch("docker_compose_mcp.subprocess.run")
    def test_start_compose_failure(self, mock_run, mock_docker, test_project_dir):
        """Test Docker Compose start failure."""
        mock_docker.return_value = MagicMock()
        mock_run.side_effect = subprocess.CalledProcessError(1, "docker compose", stderr="Error")

        manager = DockerComposeManager(test_project_dir)
        result = manager.start_compose()

        assert result is False

    @patch("docker_compose_mcp.docker.from_env")
    @patch("docker_compose_mcp.subprocess.run")
    def test_stop_compose_success(self, mock_run, mock_docker, test_project_dir):
        """Test successful Docker Compose stop."""
        mock_docker.return_value = MagicMock()
        mock_run.return_value = MagicMock(stdout="Services stopped", stderr="")

        manager = DockerComposeManager(test_project_dir)
        result = manager.stop_compose()

        assert result is True
        # Verify the command was called with the right arguments
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args[0] == (["docker", "compose", "down"],)
        assert call_args[1]["cwd"] == test_project_dir
        assert call_args[1]["capture_output"] is True
        assert call_args[1]["text"] is True
        assert call_args[1]["check"] is True
        assert call_args[1]["timeout"] == 60.0
        assert "env" in call_args[1]

    @patch("docker_compose_mcp.docker.from_env")
    @patch("docker_compose_mcp.subprocess.run")
    def test_get_compose_status_success(self, mock_run, mock_docker, test_project_dir):
        """Test successful compose status retrieval."""
        mock_docker_client = MagicMock()
        mock_docker.return_value = mock_docker_client

        # Mock docker-compose ps output
        service_data = {"Name": "test-service", "State": "running"}
        mock_run.return_value = MagicMock(stdout=json.dumps(service_data) + "\n", stderr="")

        # Mock container list
        mock_container = MagicMock()
        mock_container.name = f"{test_project_dir.name}_test-service_1"
        mock_container.status = "running"
        mock_container.attrs = {"State": {"Health": {"Status": "healthy"}}}
        mock_container.ports = {"80/tcp": [{"HostIp": "127.0.0.1", "HostPort": "8080"}]}
        mock_docker_client.containers.list.return_value = [mock_container]

        manager = DockerComposeManager(test_project_dir)
        status = manager.get_compose_status()

        assert status["project_name"] == test_project_dir.name
        assert status["compose_file_exists"] is True
        assert len(status["services"]) == 1
        assert status["services"][0]["Name"] == "test-service"
        assert len(status["containers"]) == 1

    @patch("docker_compose_mcp.docker.from_env")
    @patch("docker_compose_mcp.subprocess.run")
    def test_get_compose_logs_success(self, mock_run, mock_docker, test_project_dir):
        """Test successful compose logs retrieval."""
        mock_docker.return_value = MagicMock()
        mock_run.return_value = MagicMock(stdout="Service logs", stderr="")

        manager = DockerComposeManager(test_project_dir)
        logs = manager.get_compose_logs()

        assert logs == "Service logs"
        # Verify the command was called with the right arguments
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args[0] == (["docker", "compose", "logs", "--tail", "100"],)
        assert call_args[1]["cwd"] == test_project_dir
        assert call_args[1]["capture_output"] is True
        assert call_args[1]["text"] is True
        assert call_args[1]["check"] is True
        assert call_args[1]["timeout"] == 60.0
        assert "env" in call_args[1]

    @patch("docker_compose_mcp.docker.from_env")
    @patch("docker_compose_mcp.subprocess.run")
    def test_get_compose_logs_with_service(self, mock_run, mock_docker, test_project_dir):
        """Test compose logs retrieval for specific service."""
        mock_docker.return_value = MagicMock()
        mock_run.return_value = MagicMock(stdout="Service logs", stderr="")

        manager = DockerComposeManager(test_project_dir)
        logs = manager.get_compose_logs("test-service", 50)

        assert logs == "Service logs"
        # Verify the command was called with the right arguments
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args[0] == (["docker", "compose", "logs", "--tail", "50", "test-service"],)
        assert call_args[1]["cwd"] == test_project_dir
        assert call_args[1]["capture_output"] is True
        assert call_args[1]["text"] is True
        assert call_args[1]["check"] is True
        assert call_args[1]["timeout"] == 60.0
        assert "env" in call_args[1]

    @patch("docker_compose_mcp.docker.from_env")
    @patch("docker_compose_mcp.subprocess.run")
    def test_restart_compose_success(self, mock_run, mock_docker, test_project_dir):
        """Test successful Docker Compose restart."""
        mock_docker.return_value = MagicMock()
        mock_run.return_value = MagicMock(stdout="Services restarted", stderr="")

        manager = DockerComposeManager(test_project_dir)
        result = manager.restart_compose()

        assert result is True
        # Verify the command was called with the right arguments
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args[0] == (["docker", "compose", "restart"],)
        assert call_args[1]["cwd"] == test_project_dir
        assert call_args[1]["capture_output"] is True
        assert call_args[1]["text"] is True
        assert call_args[1]["check"] is True
        assert call_args[1]["timeout"] == 60.0
        assert "env" in call_args[1]
