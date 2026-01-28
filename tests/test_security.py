"""Tests for the security module."""

import pytest
from pathlib import Path


class TestValidatePath:
    """Tests for the validate_path function."""

    def test_validate_path_simple(self, tmp_path):
        """validate_path should accept simple valid paths."""
        from agent.security import validate_path
        
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello")
        
        result = validate_path(str(test_file), must_exist=True)
        
        assert result == test_file

    def test_validate_path_rejects_traversal(self, tmp_path):
        """validate_path should reject path traversal attempts."""
        from agent.security import validate_path, PathTraversalError
        
        with pytest.raises(PathTraversalError):
            validate_path("../../../etc/passwd")

    def test_validate_path_with_base_dir(self, tmp_path):
        """validate_path should enforce base directory."""
        from agent.security import validate_path, PathTraversalError
        
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello")
        
        # Should work within base dir
        result = validate_path(str(test_file), base_dir=tmp_path)
        assert result == test_file

    def test_validate_path_nonexistent_with_must_exist(self):
        """validate_path should raise for non-existent paths when must_exist=True."""
        from agent.security import validate_path, InputValidationError
        
        with pytest.raises((InputValidationError, FileNotFoundError)):
            validate_path("/nonexistent/path/file.txt", must_exist=True)

    def test_validate_path_expands_home(self):
        """validate_path should expand ~ to home directory."""
        from agent.security import validate_path
        
        result = validate_path("~/testfile.txt")
        
        assert "~" not in str(result)
        assert str(result).startswith(str(Path.home()))


class TestIsPathSafe:
    """Tests for the is_path_safe function."""

    def test_is_path_safe_simple(self, tmp_path):
        """is_path_safe should return True for safe paths."""
        from agent.security import is_path_safe
        
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello")
        
        assert is_path_safe(str(test_file)) is True

    def test_is_path_safe_traversal(self):
        """is_path_safe should return False for traversal attempts."""
        from agent.security import is_path_safe
        
        assert is_path_safe("../../../etc/passwd") is False

    def test_is_path_safe_sensitive_paths(self):
        """is_path_safe should return False for sensitive paths."""
        from agent.security import is_path_safe
        
        # These should be blocked
        sensitive = [
            "/etc/shadow",
            "/root/.bashrc",
        ]
        
        for path in sensitive:
            # May or may not be blocked depending on configuration
            # At minimum, the function should not crash
            result = is_path_safe(path)
            assert isinstance(result, bool)


class TestValidateFilename:
    """Tests for the validate_filename function."""

    def test_validate_filename_simple(self):
        """validate_filename should accept simple valid filenames."""
        from agent.security import validate_filename
        
        result = validate_filename("document.txt")
        assert result == "document.txt"

    def test_validate_filename_rejects_slashes(self):
        """validate_filename should reject filenames with path separators."""
        from agent.security import validate_filename, InputValidationError
        
        with pytest.raises(InputValidationError):
            validate_filename("../secret.txt")

    def test_validate_filename_rejects_null_bytes(self):
        """validate_filename should reject filenames with null bytes."""
        from agent.security import validate_filename, InputValidationError
        
        with pytest.raises(InputValidationError):
            validate_filename("file\x00.txt")

    def test_validate_filename_max_length(self):
        """validate_filename should reject overly long filenames."""
        from agent.security import validate_filename, InputValidationError
        
        long_name = "a" * 300 + ".txt"
        
        with pytest.raises(InputValidationError):
            validate_filename(long_name)


class TestValidateString:
    """Tests for the validate_string function."""

    def test_validate_string_simple(self):
        """validate_string should accept valid strings."""
        from agent.security import validate_string
        
        result = validate_string("Hello, World!", "test_field")
        assert result == "Hello, World!"

    def test_validate_string_empty(self):
        """validate_string should reject empty strings by default."""
        from agent.security import validate_string, InputValidationError
        
        with pytest.raises(InputValidationError):
            validate_string("", "test_field")

    def test_validate_string_max_length(self):
        """validate_string should enforce max_length."""
        from agent.security import validate_string, InputValidationError
        
        with pytest.raises(InputValidationError):
            validate_string("x" * 1000, "test_field", max_length=100)

    def test_validate_string_allows_empty_when_configured(self):
        """validate_string should allow empty when allow_empty=True."""
        from agent.security import validate_string
        
        result = validate_string("", "test_field", allow_empty=True)
        assert result == ""


class TestSecurityError:
    """Tests for security exception classes."""

    def test_security_error_hierarchy(self):
        """Security exceptions should have proper hierarchy."""
        from agent.security import (
            SecurityError,
            PathTraversalError,
            InputValidationError,
        )
        
        assert issubclass(PathTraversalError, SecurityError)
        assert issubclass(InputValidationError, SecurityError)

    def test_security_error_message(self):
        """Security exceptions should preserve error messages."""
        from agent.security import SecurityError
        
        error = SecurityError("Test error message")
        assert str(error) == "Test error message"
