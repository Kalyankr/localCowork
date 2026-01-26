"""Unit tests for json_tools module."""

import pytest
import json
from pathlib import Path

from agent.tools import json_tools


class TestReadWriteJson:
    """Tests for JSON read/write operations."""
    
    def test_write_and_read_json(self, tmp_path: Path):
        """Should write and read JSON correctly."""
        file_path = tmp_path / "test.json"
        data = {"name": "test", "value": 42, "items": [1, 2, 3]}
        
        json_tools.write_json(str(file_path), data)
        result = json_tools.read_json(str(file_path))
        
        assert result == data
    
    def test_write_creates_parent_dirs(self, tmp_path: Path):
        """Should create parent directories when writing."""
        file_path = tmp_path / "nested" / "dir" / "test.json"
        
        json_tools.write_json(str(file_path), {"key": "value"})
        
        assert file_path.exists()


class TestParseJson:
    """Tests for JSON parsing."""
    
    def test_parse_valid_json(self):
        """Should parse valid JSON string."""
        result = json_tools.parse_json('{"name": "test"}')
        assert result == {"name": "test"}
    
    def test_parse_invalid_json(self):
        """Should raise for invalid JSON."""
        with pytest.raises(json.JSONDecodeError):
            json_tools.parse_json("not valid json")


class TestQueryJson:
    """Tests for JSON querying."""
    
    def test_query_simple_key(self):
        """Should query simple key."""
        data = {"name": "Alice", "age": 30}
        assert json_tools.query_json(data, "name") == "Alice"
    
    def test_query_nested_key(self):
        """Should query nested keys with dot notation."""
        data = {"user": {"name": "Bob", "profile": {"email": "bob@test.com"}}}
        assert json_tools.query_json(data, "user.name") == "Bob"
        assert json_tools.query_json(data, "user.profile.email") == "bob@test.com"
    
    def test_query_array_index(self):
        """Should query array indices."""
        data = {"items": ["a", "b", "c"]}
        assert json_tools.query_json(data, "items[0]") == "a"
        assert json_tools.query_json(data, "items[2]") == "c"
    
    def test_query_combined(self):
        """Should handle combined dot and array notation."""
        data = {"users": [{"name": "Alice"}, {"name": "Bob"}]}
        assert json_tools.query_json(data, "users[0].name") == "Alice"
    
    def test_query_missing_key(self):
        """Should return None for missing key."""
        data = {"name": "test"}
        assert json_tools.query_json(data, "missing") is None


class TestFilterJson:
    """Tests for JSON filtering."""
    
    def test_filter_by_value(self):
        """Should filter list by key-value match."""
        data = [
            {"status": "active", "name": "Alice"},
            {"status": "inactive", "name": "Bob"},
            {"status": "active", "name": "Charlie"},
        ]
        
        result = json_tools.filter_json(data, "status", "active")
        
        assert len(result) == 2
        assert all(item["status"] == "active" for item in result)
    
    def test_filter_non_list(self):
        """Should raise ValueError for non-list input."""
        with pytest.raises(ValueError):
            json_tools.filter_json({"key": "value"}, "key", "value")


class TestMapJson:
    """Tests for JSON mapping."""
    
    def test_map_extract_keys(self):
        """Should extract specified keys from objects."""
        data = [
            {"id": 1, "name": "Alice", "email": "alice@test.com"},
            {"id": 2, "name": "Bob", "email": "bob@test.com"},
        ]
        
        result = json_tools.map_json(data, ["name", "email"])
        
        assert len(result) == 2
        assert result[0] == {"name": "Alice", "email": "alice@test.com"}
        assert "id" not in result[0]


class TestFlattenJson:
    """Tests for JSON flattening."""
    
    def test_flatten_nested(self):
        """Should flatten nested objects."""
        data = {"a": {"b": {"c": 1}}}
        
        result = json_tools.flatten_json(data)
        
        assert result == {"a.b.c": 1}
    
    def test_flatten_with_custom_separator(self):
        """Should use custom separator."""
        data = {"a": {"b": 1}}
        
        result = json_tools.flatten_json(data, sep="/")
        
        assert result == {"a/b": 1}


class TestDiffJson:
    """Tests for JSON diffing."""
    
    def test_diff_added_keys(self):
        """Should detect added keys."""
        obj1 = {"a": 1}
        obj2 = {"a": 1, "b": 2}
        
        result = json_tools.diff_json(obj1, obj2)
        
        assert len(result["added"]) == 1
        assert result["added"][0]["path"] == "b"
    
    def test_diff_removed_keys(self):
        """Should detect removed keys."""
        obj1 = {"a": 1, "b": 2}
        obj2 = {"a": 1}
        
        result = json_tools.diff_json(obj1, obj2)
        
        assert len(result["removed"]) == 1
        assert result["removed"][0]["path"] == "b"
    
    def test_diff_changed_values(self):
        """Should detect changed values."""
        obj1 = {"a": 1}
        obj2 = {"a": 2}
        
        result = json_tools.diff_json(obj1, obj2)
        
        assert len(result["changed"]) == 1
        assert result["changed"][0]["from"] == 1
        assert result["changed"][0]["to"] == 2


class TestDispatch:
    """Tests for dispatch function."""
    
    def test_dispatch_read(self, tmp_path: Path):
        """dispatch should route 'read' operation."""
        file_path = tmp_path / "test.json"
        file_path.write_text('{"key": "value"}')
        
        result = json_tools.dispatch("read", path=str(file_path))
        
        assert result == {"key": "value"}
    
    def test_dispatch_query(self):
        """dispatch should route 'query' operation."""
        result = json_tools.dispatch("query", data={"a": 1}, path="a")
        assert result == 1
    
    def test_dispatch_unknown_op(self):
        """dispatch should raise for unknown operations."""
        with pytest.raises(ValueError):
            json_tools.dispatch("unknown_op")
