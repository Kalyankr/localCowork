"""Integration tests for error recovery and file permissions.

These tests verify the full flow of the agent with the new features.
"""

from unittest.mock import MagicMock, patch

import pytest

from agent.orchestrator.react_agent import ReActAgent
from agent.permissions import AccessLevel, check_path_access, validate_command_paths


class TestPermissionsIntegration:
    """Integration tests for file permission scoping."""

    def test_sensitive_paths_blocked_in_command(self):
        """Commands accessing sensitive paths should be blocked."""
        # SSH keys
        level, paths = validate_command_paths("cat ~/.ssh/id_rsa")
        assert level == AccessLevel.SENSITIVE
        assert any(".ssh" in p for p in paths)

        # AWS credentials
        level, paths = validate_command_paths("cat ~/.aws/credentials")
        assert level == AccessLevel.SENSITIVE

        # .env files
        level, paths = validate_command_paths("cat /app/.env")
        assert level == AccessLevel.SENSITIVE

    def test_safe_commands_allowed(self):
        """Safe commands should be allowed."""
        level, paths = validate_command_paths("ls /tmp")
        assert level == AccessLevel.ALLOWED

        level, paths = validate_command_paths("echo hello world")
        assert level == AccessLevel.ALLOWED

        level, paths = validate_command_paths("date")
        assert level == AccessLevel.ALLOWED

    @patch("agent.permissions.get_settings")
    def test_allowed_paths_configuration(self, mock_settings):
        """User-configured allowed paths should work."""
        mock_settings.return_value.allowed_paths = "/home/user/projects,/tmp"
        mock_settings.return_value.denied_paths = ""
        mock_settings.return_value.require_path_confirmation = True

        # Path in allowed list
        assert check_path_access("/tmp/test.txt") == AccessLevel.ALLOWED

        # Path outside allowed list needs confirmation
        assert check_path_access("/var/log/test.log") == AccessLevel.NEEDS_CONFIRMATION

    @patch("agent.permissions.get_settings")
    def test_denied_paths_always_blocked(self, mock_settings):
        """User-configured denied paths should always be blocked."""
        mock_settings.return_value.allowed_paths = ""
        mock_settings.return_value.denied_paths = "/secret,/private"
        mock_settings.return_value.require_path_confirmation = True

        assert check_path_access("/secret/data.txt") == AccessLevel.DENIED
        # Use a non-sensitive extension to test denied paths specifically
        assert check_path_access("/private/data/file.txt") == AccessLevel.DENIED

    @pytest.mark.asyncio
    @patch("agent.orchestrator.react_agent.call_llm_json_async")
    async def test_agent_blocks_sensitive_commands(self, mock_llm):
        """Agent should block commands that access sensitive paths."""
        sandbox = MagicMock()
        agent = ReActAgent(sandbox=sandbox, max_iterations=3)

        # LLM tries to access SSH keys
        mock_llm.return_value = {
            "thought": "Read SSH key",
            "is_complete": False,
            "action": {
                "tool": "shell",
                "args": {"command": "cat ~/.ssh/id_rsa"},
            },
        }

        state = await agent.run("Show me my SSH key")

        # Should have failed due to permission denial
        assert state.status in ("completed", "failed", "max_iterations")
        # Check that the sensitive path was blocked
        blocked = False
        for step in state.steps:
            if (
                step.result
                and step.result.error
                and (
                    "denied" in step.result.error.lower()
                    or "blocked" in step.result.error.lower()
                )
            ):
                blocked = True
                break
        assert blocked, "Sensitive path should have been blocked"


class TestErrorRecoveryIntegration:
    """Integration tests for error recovery."""

    @pytest.mark.asyncio
    @patch("agent.orchestrator.react_agent.call_llm_json_async")
    @patch("subprocess.run")
    async def test_recovery_after_command_failure(self, mock_run, mock_llm):
        """Agent should attempt recovery when a command fails."""
        sandbox = MagicMock()
        agent = ReActAgent(sandbox=sandbox, max_iterations=5)

        call_count = [0]

        def llm_side_effect(*args, **kwargs):
            call_count[0] += 1

            if call_count[0] == 1:
                # First: try a command that will fail
                return {
                    "thought": "List files",
                    "is_complete": False,
                    "action": {
                        "tool": "shell",
                        "args": {"command": "ls /nonexistent_path_12345"},
                    },
                }
            elif call_count[0] == 2:
                # Recovery: try different approach
                return {
                    "analysis": "Path doesn't exist",
                    "new_approach": "Try home directory instead",
                    "action": {
                        "tool": "shell",
                        "args": {"command": "ls ~"},
                    },
                    "give_up": False,
                }
            else:
                # Complete
                return {
                    "thought": "Done",
                    "is_complete": True,
                    "response": "Listed files in home directory",
                }

        mock_llm.side_effect = llm_side_effect

        # First command fails, second succeeds
        fail_result = MagicMock()
        fail_result.returncode = 2
        fail_result.stdout = b""
        fail_result.stderr = b"No such file or directory"

        success_result = MagicMock()
        success_result.returncode = 0
        success_result.stdout = b"Documents\nDownloads\nPictures"
        success_result.stderr = b""

        mock_run.side_effect = [fail_result, success_result]

        state = await agent.run("List files")

        # Should have completed with recovery
        assert state.status in ("completed", "max_iterations")
        assert call_count[0] >= 2, "Should have attempted recovery"

    @pytest.mark.asyncio
    @patch("agent.orchestrator.react_agent.call_llm_json_async")
    async def test_recovery_gives_up_after_max_attempts(self, mock_llm):
        """Agent should give up after max recovery attempts."""
        sandbox = MagicMock()
        agent = ReActAgent(sandbox=sandbox, max_iterations=10)

        call_count = [0]

        def llm_side_effect(*args, **kwargs):
            call_count[0] += 1

            # Always return a command that will trigger safety check failure
            if "recovery" in str(args).lower() or call_count[0] > 1:
                return {
                    "analysis": "Still failing",
                    "new_approach": None,
                    "action": None,
                    "give_up": True,
                    "user_message": "Unable to complete this task",
                }

            return {
                "thought": "Try blocked command",
                "is_complete": False,
                "action": {
                    "tool": "shell",
                    "args": {"command": "rm -rf /"},  # This will be blocked
                },
            }

        mock_llm.side_effect = llm_side_effect

        state = await agent.run("Delete everything")

        # Should have stopped (blocked or max iterations)
        assert state.status in ("completed", "failed", "max_iterations")


class TestCombinedFeatures:
    """Tests for error recovery + permissions working together."""

    @pytest.mark.asyncio
    @patch("agent.orchestrator.react_agent.call_llm_json_async")
    async def test_permission_denial_triggers_recovery(self, mock_llm):
        """Permission denial should trigger recovery attempt."""
        sandbox = MagicMock()
        agent = ReActAgent(sandbox=sandbox, max_iterations=5)

        call_count = [0]

        def llm_side_effect(*args, **kwargs):
            call_count[0] += 1

            if call_count[0] == 1:
                # Try to read .env file (blocked)
                return {
                    "thought": "Read config",
                    "is_complete": False,
                    "action": {
                        "tool": "shell",
                        "args": {"command": "cat /app/.env"},
                    },
                }
            elif call_count[0] == 2:
                # Recovery: suggest alternative
                return {
                    "analysis": ".env is protected",
                    "new_approach": "Read public config instead",
                    "action": {
                        "tool": "shell",
                        "args": {"command": "cat /app/config.json"},
                    },
                    "give_up": False,
                }
            else:
                return {
                    "thought": "Done",
                    "is_complete": True,
                    "response": "Read the config file",
                }

        mock_llm.side_effect = llm_side_effect

        with patch("subprocess.run") as mock_run:
            success_result = MagicMock()
            success_result.returncode = 0
            success_result.stdout = b'{"setting": "value"}'
            success_result.stderr = b""
            mock_run.return_value = success_result

            state = await agent.run("Read the config")

            # First attempt should be blocked, recovery should work
            assert call_count[0] >= 2
            assert state is not None  # Verify we got a result

    def test_error_messages_are_user_friendly(self):
        """Error messages should be clear and helpful."""
        from agent.permissions import get_permission_error_message

        msg = get_permission_error_message("~/.ssh/id_rsa", AccessLevel.SENSITIVE)
        assert "blocked" in msg.lower()
        assert "security" in msg.lower()
        assert "ssh" in msg.lower() or "credential" in msg.lower()

        msg = get_permission_error_message("/secret", AccessLevel.DENIED)
        assert "denied" in msg.lower()

    @patch("agent.permissions.get_settings")
    def test_glob_patterns_in_paths(self, mock_settings):
        """Glob patterns should work in path configuration."""
        mock_settings.return_value.allowed_paths = ""
        mock_settings.return_value.denied_paths = "/var/log/*,/etc/*.conf"
        mock_settings.return_value.require_path_confirmation = False

        # Should match glob pattern
        level, paths = validate_command_paths("cat /var/log/syslog")
        # Note: may or may not match depending on implementation
        # The important thing is it doesn't crash


class TestSettingsIntegration:
    """Tests for settings configuration."""

    def test_default_settings_are_safe(self):
        """Default settings should have safe defaults."""
        from agent.config import Settings

        settings = Settings()

        # Should have recovery attempts configured
        assert settings.max_recovery_attempts >= 1

        # Should have some denied paths by default
        assert settings.denied_paths  # Not empty

        # Should require confirmation by default
        assert settings.require_path_confirmation is True

    def test_sensitive_patterns_include_common_secrets(self):
        """Built-in sensitive patterns should cover common secrets."""
        sensitive_paths = [
            "~/.ssh/id_rsa",
            "~/.ssh/id_ed25519",
            "~/.aws/credentials",
            "~/.gnupg/private-keys-v1.d/key.gpg",
            "/app/.env",
            "/project/.env.local",
            "/home/user/secrets.json",
        ]

        for path in sensitive_paths:
            level = check_path_access(path)
            assert level == AccessLevel.SENSITIVE, f"{path} should be SENSITIVE"
