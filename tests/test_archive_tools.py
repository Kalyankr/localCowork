"""Unit tests for archive_tools module."""

import pytest
from pathlib import Path
import zipfile
import tarfile

from agent.tools import archive_tools


class TestCreateZip:
    """Tests for zip creation."""
    
    def test_create_zip_from_file(self, tmp_path: Path):
        """Should create a ZIP from a single file."""
        src_file = tmp_path / "test.txt"
        src_file.write_text("hello world")
        dest = tmp_path / "archive.zip"
        
        result = archive_tools.create_zip(str(src_file), str(dest))
        
        assert "Created" in result
        assert dest.exists()
        with zipfile.ZipFile(dest, 'r') as zf:
            assert "test.txt" in zf.namelist()
    
    def test_create_zip_from_directory(self, tmp_path: Path):
        """Should create a ZIP from a directory."""
        src_dir = tmp_path / "source"
        src_dir.mkdir()
        (src_dir / "file1.txt").write_text("content1")
        (src_dir / "file2.txt").write_text("content2")
        dest = tmp_path / "archive.zip"
        
        result = archive_tools.create_zip(str(src_dir), str(dest))
        
        assert dest.exists()
        with zipfile.ZipFile(dest, 'r') as zf:
            names = zf.namelist()
            assert any("file1.txt" in name for name in names)
            assert any("file2.txt" in name for name in names)


class TestExtractZip:
    """Tests for zip extraction."""
    
    def test_extract_zip(self, tmp_path: Path):
        """Should extract a ZIP archive."""
        # Create a zip first
        zip_path = tmp_path / "test.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("extracted.txt", "content")
        
        dest_dir = tmp_path / "extracted"
        result = archive_tools.extract_zip(str(zip_path), str(dest_dir))
        
        assert "Extracted" in result
        assert (dest_dir / "extracted.txt").exists()
        assert (dest_dir / "extracted.txt").read_text() == "content"


class TestListZip:
    """Tests for listing zip contents."""
    
    def test_list_zip_contents(self, tmp_path: Path):
        """Should list contents of a ZIP archive."""
        zip_path = tmp_path / "test.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("file1.txt", "a")
            zf.writestr("file2.txt", "bb")
        
        result = archive_tools.list_zip(str(zip_path))
        
        assert isinstance(result, list)
        assert len(result) == 2
        names = [item["name"] for item in result]
        assert "file1.txt" in names
        assert "file2.txt" in names


class TestCreateTar:
    """Tests for tar creation."""
    
    def test_create_tar_gz(self, tmp_path: Path):
        """Should create a gzipped TAR archive."""
        src_file = tmp_path / "test.txt"
        src_file.write_text("content")
        dest = tmp_path / "archive.tar.gz"
        
        result = archive_tools.create_tar(str(src_file), str(dest))
        
        assert "Created" in result
        assert dest.exists()
        with tarfile.open(dest, 'r:gz') as tf:
            names = tf.getnames()
            assert "test.txt" in names


class TestExtractAuto:
    """Tests for auto-extraction."""
    
    def test_extract_auto_zip(self, tmp_path: Path):
        """Should auto-detect and extract ZIP."""
        zip_path = tmp_path / "test.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("auto.txt", "content")
        
        dest = tmp_path / "dest"
        result = archive_tools.extract_auto(str(zip_path), str(dest))
        
        assert (dest / "auto.txt").exists()
    
    def test_extract_auto_tar(self, tmp_path: Path):
        """Should auto-detect and extract TAR.GZ."""
        # Create tar.gz
        tar_path = tmp_path / "test.tar.gz"
        src_file = tmp_path / "src.txt"
        src_file.write_text("content")
        
        with tarfile.open(tar_path, 'w:gz') as tf:
            tf.add(src_file, arcname="src.txt")
        
        dest = tmp_path / "dest"
        result = archive_tools.extract_auto(str(tar_path), str(dest))
        
        assert (dest / "src.txt").exists()


class TestDispatch:
    """Tests for dispatch function."""
    
    def test_dispatch_zip(self, tmp_path: Path):
        """dispatch should route 'zip' operation."""
        src = tmp_path / "file.txt"
        src.write_text("test")
        dest = tmp_path / "out.zip"
        
        result = archive_tools.dispatch("zip", source=str(src), dest=str(dest))
        
        assert "Created" in result
        assert dest.exists()
    
    def test_dispatch_unknown_op(self):
        """dispatch should raise for unknown operations."""
        with pytest.raises(ValueError):
            archive_tools.dispatch("unknown_op", source="/tmp", dest="/tmp")
