"""Unit tests for file_tools module."""

import pytest
from pathlib import Path
import tempfile
import shutil

from agent.tools import file_tools


class TestListFiles:
    """Tests for list_files function."""
    
    def test_list_files_returns_list(self, tmp_path: Path):
        """list_files should return a list of dicts."""
        # Create test files
        (tmp_path / "file1.txt").write_text("content")
        (tmp_path / "file2.py").write_text("print('hello')")
        
        result = file_tools.list_files(str(tmp_path))
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(item, dict) for item in result)
    
    def test_list_files_contains_metadata(self, tmp_path: Path):
        """Each file dict should contain required metadata keys."""
        (tmp_path / "test.txt").write_text("hello")
        
        result = file_tools.list_files(str(tmp_path))
        
        assert len(result) == 1
        file_info = result[0]
        assert "path" in file_info
        assert "name" in file_info
        assert "size" in file_info
        assert "mtime" in file_info
        assert "is_dir" in file_info
        assert file_info["name"] == "test.txt"
    
    def test_list_files_nonexistent_path(self):
        """list_files should return empty list for nonexistent path."""
        result = file_tools.list_files("/nonexistent/path/12345")
        assert result == []
    
    def test_list_files_expands_tilde(self):
        """list_files should expand ~ to home directory."""
        result = file_tools.list_files("~")
        assert isinstance(result, list)


class TestMoveFile:
    """Tests for move_file function."""
    
    def test_move_single_file(self, tmp_path: Path):
        """Should move a single file to destination."""
        src_file = tmp_path / "source.txt"
        src_file.write_text("content")
        dest_dir = tmp_path / "dest"
        
        result = file_tools.move_file(str(src_file), str(dest_dir))
        
        assert "Moved" in result
        assert (dest_dir / "source.txt").exists()
        assert not src_file.exists()
    
    def test_move_empty_list(self):
        """Should return message for empty list."""
        result = file_tools.move_file([], "/some/dest")
        assert "No files found" in result
    
    def test_move_creates_dest_directory(self, tmp_path: Path):
        """Should create destination directory if it doesn't exist."""
        src_file = tmp_path / "file.txt"
        src_file.write_text("hello")
        dest_dir = tmp_path / "new" / "nested" / "dir"
        
        result = file_tools.move_file(str(src_file), str(dest_dir))
        
        assert dest_dir.exists()


class TestCreateDir:
    """Tests for create_dir function."""
    
    def test_create_simple_dir(self, tmp_path: Path):
        """Should create a simple directory."""
        new_dir = tmp_path / "newdir"
        
        result = file_tools.create_dir(str(new_dir))
        
        assert "Created" in result
        assert new_dir.exists()
        assert new_dir.is_dir()
    
    def test_create_nested_dirs(self, tmp_path: Path):
        """Should create nested directories."""
        nested = tmp_path / "a" / "b" / "c"
        
        result = file_tools.create_dir(str(nested))
        
        assert nested.exists()


class TestReadWrite:
    """Tests for read_text and write_text functions."""
    
    def test_write_and_read(self, tmp_path: Path):
        """Should write and read text correctly."""
        file_path = tmp_path / "test.txt"
        content = "Hello, World!"
        
        file_tools.write_text(str(file_path), content)
        result = file_tools.read_text(str(file_path))
        
        assert result == content
    
    def test_read_nonexistent_file(self, tmp_path: Path):
        """Should raise FileNotFoundError for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            file_tools.read_text(str(tmp_path / "nonexistent.txt"))


class TestDispatch:
    """Tests for dispatch function."""
    
    def test_dispatch_list(self, tmp_path: Path):
        """dispatch should route 'list' operation correctly."""
        (tmp_path / "file.txt").touch()
        
        result = file_tools.dispatch("list", path=str(tmp_path))
        
        assert isinstance(result, list)
        assert len(result) == 1
    
    def test_dispatch_mkdir(self, tmp_path: Path):
        """dispatch should route 'mkdir' operation correctly."""
        new_dir = tmp_path / "created"
        
        result = file_tools.dispatch("mkdir", path=str(new_dir))
        
        assert "Created" in result
        assert new_dir.exists()
    
    def test_dispatch_unknown_op(self):
        """dispatch should raise for unknown operations."""
        with pytest.raises((ValueError, KeyError)):
            file_tools.dispatch("unknown_op", path="/tmp")
