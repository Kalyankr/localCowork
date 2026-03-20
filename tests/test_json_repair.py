"""Extended tests for the JSON repair engine in agent.llm.client."""

import pytest

from agent.llm.client import repair_json


class TestRepairJSONEdgeCases:
    """Edge-case tests for repair_json beyond the basics in test_llm_client.py."""

    # --- Nested / complex structures ---

    def test_nested_objects(self):
        text = '{"a": {"b": {"c": 1}}}'
        assert repair_json(text) == {"a": {"b": {"c": 1}}}

    def test_nested_arrays(self):
        text = '{"items": [[1, 2], [3, 4]]}'
        result = repair_json(text)
        assert result["items"] == [[1, 2], [3, 4]]

    def test_object_with_array_value(self):
        text = '{"tags": ["python", "asyncio", "fastapi"]}'
        assert repair_json(text)["tags"] == ["python", "asyncio", "fastapi"]

    # --- Trailing commas ---

    def test_trailing_comma_in_object(self):
        text = '{"a": 1, "b": 2,}'
        result = repair_json(text)
        assert result == {"a": 1, "b": 2}

    def test_trailing_comma_in_array(self):
        text = '{"items": [1, 2, 3,]}'
        result = repair_json(text)
        assert result["items"] == [1, 2, 3]

    # --- Single-quote handling ---

    def test_single_quoted_values(self):
        """repair_json handles single-quoted values with double-quoted keys."""
        text = """{"key": 'value', "status": 'ok'}"""
        result = repair_json(text)
        assert result == {"key": "value", "status": "ok"}

    # --- Markdown wrapping variants ---

    def test_markdown_with_language_tag(self):
        text = '```json\n{"ok": true}\n```'
        assert repair_json(text) == {"ok": True}

    def test_markdown_without_language_tag(self):
        text = '```\n{"ok": true}\n```'
        assert repair_json(text) == {"ok": True}

    def test_triple_backtick_with_extra_whitespace(self):
        text = '```json  \n  {"ok": true}  \n  ```  '
        assert repair_json(text) == {"ok": True}

    # --- Literal newlines inside strings ---

    def test_literal_newline_inside_string_value(self):
        text = '{"msg": "line one\nline two"}'
        result = repair_json(text)
        assert "line one" in result["msg"]
        assert "line two" in result["msg"]

    def test_literal_tab_inside_string_value(self):
        text = '{"msg": "col1\tcol2"}'
        result = repair_json(text)
        assert "col1" in result["msg"]

    # --- Unicode content ---

    def test_unicode_characters(self):
        text = '{"greeting": "こんにちは", "emoji": "🚀"}'
        result = repair_json(text)
        assert result["greeting"] == "こんにちは"
        assert result["emoji"] == "🚀"

    # --- Surrounding junk text ---

    def test_json_buried_in_prose(self):
        text = (
            'Sure! Here is the analysis:\n\n{"thought": "analyzing", '
            '"is_complete": false, "action": {"tool": "shell"}}\n\nHope that helps!'
        )
        result = repair_json(text)
        assert result["thought"] == "analyzing"
        assert result["action"]["tool"] == "shell"

    # --- Boolean / null values ---

    def test_boolean_and_null_values(self):
        text = '{"a": true, "b": false, "c": null}'
        result = repair_json(text)
        assert result == {"a": True, "b": False, "c": None}

    # --- Empty object ---

    def test_empty_object(self):
        assert repair_json("{}") == {}

    # --- Error cases ---

    def test_no_json_raises_value_error(self):
        with pytest.raises(ValueError, match="No JSON"):
            repair_json("absolutely no json here")

    def test_plain_array_raises_value_error(self):
        """repair_json expects an object, not an array at top level."""
        with pytest.raises(ValueError, match="No JSON"):
            repair_json("[1, 2, 3]")

    # --- Numeric values ---

    def test_integer_and_float_values(self):
        text = '{"int": 42, "float": 3.14, "neg": -1}'
        result = repair_json(text)
        assert result["int"] == 42
        assert result["float"] == pytest.approx(3.14)
        assert result["neg"] == -1
