"""Unit tests for shell_tools module."""

import pytest

from agent.tools import shell_tools


class TestIsCommandSafe:
    """Tests for command safety checking."""
    
    def test_allowed_commands(self):
        """Should allow whitelisted commands."""
        safe_commands = ["ls", "find .", "grep pattern file", "head -n 10 file"]
        
        for cmd in safe_commands:
            is_safe, reason = shell_tools.is_command_safe(cmd)
            assert is_safe, f"Command '{cmd}' should be safe but got: {reason}"
    
    def test_blocked_commands(self):
        """Should block dangerous commands."""
        blocked_commands = ["rm file.txt", "sudo apt update", "chmod 777 file"]
        
        for cmd in blocked_commands:
            is_safe, reason = shell_tools.is_command_safe(cmd)
            assert not is_safe, f"Command '{cmd}' should be blocked"
    
    def test_unlisted_command(self):
        """Should block commands not in whitelist."""
        is_safe, reason = shell_tools.is_command_safe("vim file.txt")
        assert not is_safe
        assert "not in the allowed list" in reason
    
    def test_empty_command(self):
        """Should reject empty commands."""
        is_safe, reason = shell_tools.is_command_safe("")
        assert not is_safe


class TestRunCommand:
    """Tests for command execution."""
    
    def test_run_simple_command(self):
        """Should run a simple allowed command."""
        result = shell_tools.run_command("echo hello")
        
        assert "stdout" in result
        assert "hello" in result["stdout"]
        assert result["returncode"] == 0
    
    def test_run_blocked_command(self):
        """Should reject blocked commands."""
        result = shell_tools.run_command("rm -rf /")
        
        assert "error" in result
        assert "not allowed" in result["error"]
    
    def test_run_with_cwd(self, tmp_path):
        """Should run command in specified directory."""
        result = shell_tools.run_command("pwd", cwd=str(tmp_path))
        
        assert str(tmp_path) in result["stdout"]
    
    def test_run_bypass_safety_check(self):
        """Should allow bypassing safety check (for internal use)."""
        result = shell_tools.run_command("echo test", check_safety=False)
        
        assert result["returncode"] == 0


class TestGetSystemInfo:
    """Tests for system info retrieval."""
    
    def test_sysinfo_returns_dict(self):
        """Should return a dictionary with system info."""
        result = shell_tools.get_system_info()
        
        assert isinstance(result, dict)
        assert "platform" in result
        assert "python_version" in result
        assert "hostname" in result


class TestDispatch:
    """Tests for dispatch function."""
    
    def test_dispatch_run(self):
        """dispatch should route 'run' operation."""
        result = shell_tools.dispatch("run", cmd="echo test")
        
        assert "stdout" in result
        assert "test" in result["stdout"]
    
    def test_dispatch_sysinfo(self):
        """dispatch should route 'sysinfo' operation."""
        result = shell_tools.dispatch("sysinfo")
        
        assert "platform" in result
    
    def test_dispatch_allowed(self):
        """dispatch should route 'allowed' operation."""
        result = shell_tools.dispatch("allowed")
        
        assert "allowed_commands" in result
        assert "ls" in result["allowed_commands"]
    
    def test_dispatch_unknown_op(self):
        """dispatch should raise for unknown operations."""
        with pytest.raises(ValueError):
            shell_tools.dispatch("unknown_op")
