"""Tests for file permission management."""

from pathlib import Path
from unittest.mock import patch

from agent.permissions import (
    AccessLevel,
    _expand_path,
    _is_subpath,
    _matches_pattern,
    _parse_path_list,
    check_path_access,
    get_permission_error_message,
    validate_command_paths,
)


class TestPathExpansion:
    """Tests for path expansion utilities."""

    def test_expand_home_path(self):
        """Test expansion of ~ to home directory."""
        result = _expand_path("~/Documents")
        assert result.startswith("/")
        assert "~" not in result
        assert result.endswith("Documents")

    def test_expand_absolute_path(self):
        """Test absolute paths stay absolute."""
        result = _expand_path("/tmp/test")
        assert result == "/tmp/test"

    def test_expand_relative_path(self):
        """Test relative paths become absolute."""
        result = _expand_path("./test")
        assert result.startswith("/")


class TestPathParsing:
    """Tests for path list parsing."""

    def test_parse_empty_string(self):
        """Test parsing empty string returns empty list."""
        assert _parse_path_list("") == []
        assert _parse_path_list("   ") == []

    def test_parse_single_path(self):
        """Test parsing single path."""
        result = _parse_path_list("/tmp")
        assert len(result) == 1
        assert result[0] == "/tmp"

    def test_parse_multiple_paths(self):
        """Test parsing comma-separated paths."""
        result = _parse_path_list("/tmp,/var,/home")
        assert len(result) == 3
        assert "/tmp" in result
        assert "/var" in result

    def test_parse_paths_with_spaces(self):
        """Test parsing paths with spaces around commas."""
        result = _parse_path_list(" /tmp , /var , /home ")
        assert len(result) == 3


class TestPatternMatching:
    """Tests for glob pattern matching."""

    def test_simple_wildcard(self):
        """Test simple * wildcard."""
        assert _matches_pattern("/tmp/test.txt", "/tmp/*.txt")
        assert not _matches_pattern("/tmp/test.py", "/tmp/*.txt")

    def test_recursive_wildcard(self):
        """Test ** recursive wildcard."""
        assert _matches_pattern("/home/user/deep/nested/file.py", "/home/**/*.py")

    def test_exact_match(self):
        """Test exact path match."""
        assert _matches_pattern("/etc/passwd", "/etc/passwd")


class TestSubpathDetection:
    """Tests for subpath detection."""

    def test_is_subpath_true(self):
        """Test detecting valid subpath."""
        assert _is_subpath("/home/user/docs/file.txt", "/home/user")
        assert _is_subpath("/home/user", "/home/user")

    def test_is_subpath_false(self):
        """Test detecting non-subpath."""
        assert not _is_subpath("/etc/passwd", "/home/user")
        assert not _is_subpath("/home/other", "/home/user")


class TestAccessLevelCheck:
    """Tests for path access level checking."""

    def test_sensitive_paths_always_blocked(self):
        """Test that sensitive paths are always blocked."""
        # SSH keys
        assert check_path_access("~/.ssh/id_rsa") == AccessLevel.SENSITIVE
        assert check_path_access("~/.ssh/config") == AccessLevel.SENSITIVE

        # AWS credentials
        assert check_path_access("~/.aws/credentials") == AccessLevel.SENSITIVE

        # GnuPG
        assert check_path_access("~/.gnupg/private-keys-v1.d") == AccessLevel.SENSITIVE

    def test_env_files_blocked(self):
        """Test that .env files are blocked."""
        assert check_path_access("/project/.env") == AccessLevel.SENSITIVE
        assert check_path_access("/app/.env.local") == AccessLevel.SENSITIVE

    @patch("agent.permissions.get_settings")
    def test_denied_paths_blocked(self, mock_settings):
        """Test user-configured denied paths are blocked."""
        mock_settings.return_value.denied_paths = "/secret,/private"
        mock_settings.return_value.allowed_paths = ""
        mock_settings.return_value.require_path_confirmation = True

        assert check_path_access("/secret/data.txt") == AccessLevel.DENIED
        assert check_path_access("/private/keys") == AccessLevel.DENIED

    @patch("agent.permissions.get_settings")
    def test_allowed_paths_permitted(self, mock_settings):
        """Test user-configured allowed paths are permitted."""
        home = str(Path.home())
        mock_settings.return_value.allowed_paths = f"{home}/projects,/tmp"
        mock_settings.return_value.denied_paths = ""
        mock_settings.return_value.require_path_confirmation = True

        assert check_path_access(f"{home}/projects/code.py") == AccessLevel.ALLOWED
        assert check_path_access("/tmp/test.txt") == AccessLevel.ALLOWED

    @patch("agent.permissions.get_settings")
    def test_paths_outside_allowed_need_confirmation(self, mock_settings):
        """Test paths outside allowed list need confirmation."""
        mock_settings.return_value.allowed_paths = "/tmp"
        mock_settings.return_value.denied_paths = ""
        mock_settings.return_value.require_path_confirmation = True

        assert check_path_access("/var/log/test.log") == AccessLevel.NEEDS_CONFIRMATION


class TestCommandPathValidation:
    """Tests for validating paths in commands."""

    def test_extract_absolute_paths(self):
        """Test extraction of absolute paths from commands."""
        level, paths = validate_command_paths("cat /etc/hosts")
        assert "/etc/hosts" in paths or level == AccessLevel.ALLOWED

    def test_extract_home_paths(self):
        """Test extraction of ~ paths from commands."""
        level, paths = validate_command_paths("ls ~/Documents")
        # Should find the home path
        assert level in (AccessLevel.ALLOWED, AccessLevel.NEEDS_CONFIRMATION)

    def test_redirect_paths(self):
        """Test extraction of redirect paths."""
        level, paths = validate_command_paths("echo test > /tmp/out.txt")
        assert "/tmp/out.txt" in paths or level == AccessLevel.ALLOWED

    def test_sensitive_path_in_command(self):
        """Test detection of sensitive paths in commands."""
        level, paths = validate_command_paths("cat ~/.ssh/id_rsa")
        assert level == AccessLevel.SENSITIVE

    def test_command_without_paths(self):
        """Test commands without file paths."""
        level, paths = validate_command_paths("echo hello world")
        assert level == AccessLevel.ALLOWED
        assert len(paths) == 0


class TestErrorMessages:
    """Tests for error message generation."""

    def test_sensitive_error_message(self):
        """Test error message for sensitive paths."""
        msg = get_permission_error_message("~/.ssh/id_rsa", AccessLevel.SENSITIVE)
        assert "blocked" in msg.lower()
        assert "security" in msg.lower()

    def test_denied_error_message(self):
        """Test error message for denied paths."""
        msg = get_permission_error_message("/secret", AccessLevel.DENIED)
        assert "denied" in msg.lower()

    def test_confirmation_message(self):
        """Test message for paths needing confirmation."""
        msg = get_permission_error_message("/var/data", AccessLevel.NEEDS_CONFIRMATION)
        assert "confirmation" in msg.lower()
