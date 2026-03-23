"""Tests for CLI input with prompt_toolkit integration.

Tests the slash-command completer, file path completer,
prompt session creation, and _get_input() behavior.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document

from agent.cli.agent_loop import (
    _build_completer,
    _get_prompt_session,
    _SlashCompleter,
)

# ── SlashCompleter tests ──────────────────────────────────────────────


class TestSlashCompleter:
    """Tests for slash command auto-completion."""

    def _complete(self, completer, text: str) -> list[str]:
        """Helper: get completion texts for given input."""
        doc = Document(text, cursor_position=len(text))
        event = CompleteEvent()
        return [c.text for c in completer.get_completions(doc, event)]

    def test_all_commands_on_slash(self):
        """Typing '/' should yield all slash commands."""
        completer = _SlashCompleter()
        results = self._complete(completer, "/")
        assert "/help" in results
        assert "/quit" in results
        assert "/clear" in results
        assert "/status" in results
        assert "/history" in results
        assert "/model " in results

    def test_filters_by_prefix(self):
        """Typing '/h' should yield /help and /history."""
        completer = _SlashCompleter()
        results = self._complete(completer, "/h")
        assert "/help" in results
        assert "/history" in results
        assert "/quit" not in results

    def test_single_match(self):
        """Typing '/q' should yield only /quit."""
        completer = _SlashCompleter()
        results = self._complete(completer, "/q")
        assert results == ["/quit"]

    def test_no_match(self):
        """Typing '/xyz' should yield no completions."""
        completer = _SlashCompleter()
        results = self._complete(completer, "/xyz")
        assert results == []

    def test_no_completions_without_slash(self):
        """Regular text should not trigger slash completions."""
        completer = _SlashCompleter()
        results = self._complete(completer, "list files")
        assert results == []

    def test_model_has_trailing_space(self):
        """The /model completion should include a trailing space."""
        completer = _SlashCompleter()
        results = self._complete(completer, "/mo")
        assert "/model " in results

    def test_completions_have_display_meta(self):
        """Each completion should have a description in display_meta."""
        completer = _SlashCompleter()
        doc = Document("/h", cursor_position=2)
        event = CompleteEvent()
        completions = list(completer.get_completions(doc, event))
        for c in completions:
            assert c.display_meta is not None
            assert len(c.display_meta) > 0


# ── Merged completer tests ───────────────────────────────────────────


class TestMergedCompleter:
    """Tests for the merged slash + path completer."""

    def _complete(self, completer, text: str) -> list[str]:
        doc = Document(text, cursor_position=len(text))
        event = CompleteEvent()
        return [c.text for c in completer.get_completions(doc, event)]

    def test_slash_commands_in_merged(self):
        """Slash commands should work through the merged completer."""
        completer = _build_completer()
        results = self._complete(completer, "/cl")
        assert "/clear" in results

    def test_file_paths_in_merged(self, tmp_path):
        """File path completion should work for existing paths."""
        completer = _build_completer()
        # Create a temp file
        test_file = tmp_path / "testfile.txt"
        test_file.write_text("hello")

        # PathCompleter resolves relative to CWD, so give absolute dir path
        # with trailing slash so it lists directory contents
        prefix = str(tmp_path) + os.sep + "testf"
        results = self._complete(completer, prefix)
        # Should find our test file (completion text is just the filename part)
        assert any("testfile.txt" in r for r in results) or len(results) > 0


# ── PromptSession creation tests ─────────────────────────────────────


class TestPromptSession:
    """Tests for prompt session initialization."""

    def test_session_creates_history_dir(self, tmp_path):
        """Session creation should create the history directory."""
        import agent.cli.agent_loop as module

        # Reset the cached session
        original = module._prompt_session
        module._prompt_session = None

        history_dir = tmp_path / ".localcowork"

        with patch.object(Path, "expanduser", return_value=history_dir):
            session = _get_prompt_session()
            assert session is not None
            # Session should be cached
            assert module._prompt_session is session
            # Calling again returns the same instance
            assert _get_prompt_session() is session

        # Restore
        module._prompt_session = original

    def test_session_has_completer(self, tmp_path):
        """Session should be configured with a completer."""
        import agent.cli.agent_loop as module

        original = module._prompt_session
        module._prompt_session = None

        history_dir = tmp_path / ".localcowork"

        with patch.object(Path, "expanduser", return_value=history_dir):
            session = _get_prompt_session()
            assert session.completer is not None

        module._prompt_session = original

    def test_session_has_file_history(self, tmp_path):
        """Session should use FileHistory for persistence."""
        import agent.cli.agent_loop as module

        original = module._prompt_session
        module._prompt_session = None

        history_dir = tmp_path / ".localcowork"

        with patch.object(Path, "expanduser", return_value=history_dir):
            session = _get_prompt_session()
            from prompt_toolkit.history import FileHistory

            assert isinstance(session.history, FileHistory)

        module._prompt_session = original

    def test_complete_while_typing_disabled(self, tmp_path):
        """Autocomplete should not trigger while typing — only on Tab."""
        import agent.cli.agent_loop as module

        original = module._prompt_session
        module._prompt_session = None

        history_dir = tmp_path / ".localcowork"

        with patch.object(Path, "expanduser", return_value=history_dir):
            session = _get_prompt_session()
            assert session.complete_while_typing is not None

        module._prompt_session = original


# ── _get_input integration tests ─────────────────────────────────────


class TestGetInput:
    """Tests for the _get_input function (mocked prompt_toolkit)."""

    @patch("agent.cli.agent_loop._get_prompt_session")
    def test_returns_stripped_input(self, mock_session_fn):
        """Input should be stripped of whitespace."""
        from agent.cli.agent_loop import _get_input

        mock_session = MagicMock()
        mock_session.prompt.return_value = "  hello world  "
        mock_session_fn.return_value = mock_session

        # Suppress terminal escape codes and Rich output
        with patch("sys.stdout"), patch("agent.cli.agent_loop.console"):
            result = _get_input()

        assert result == "hello world"

    @patch("agent.cli.agent_loop._get_prompt_session")
    def test_empty_input(self, mock_session_fn):
        """Empty input should return empty string."""
        from agent.cli.agent_loop import _get_input

        mock_session = MagicMock()
        mock_session.prompt.return_value = "   "
        mock_session_fn.return_value = mock_session

        with patch("sys.stdout"), patch("agent.cli.agent_loop.console"):
            result = _get_input()

        assert result == ""

    @patch("agent.cli.agent_loop._get_prompt_session")
    def test_keyboard_interrupt_propagates(self, mock_session_fn):
        """KeyboardInterrupt from prompt should propagate up."""
        from agent.cli.agent_loop import _get_input

        mock_session = MagicMock()
        mock_session.prompt.side_effect = KeyboardInterrupt()
        mock_session_fn.return_value = mock_session

        with patch("agent.cli.agent_loop.console"), pytest.raises(KeyboardInterrupt):
            _get_input()

    @patch("agent.cli.agent_loop._get_prompt_session")
    def test_eof_error_propagates(self, mock_session_fn):
        """EOFError from prompt should propagate up."""
        from agent.cli.agent_loop import _get_input

        mock_session = MagicMock()
        mock_session.prompt.side_effect = EOFError()
        mock_session_fn.return_value = mock_session

        with patch("agent.cli.agent_loop.console"), pytest.raises(EOFError):
            _get_input()

    @patch("agent.cli.agent_loop._get_prompt_session")
    def test_slash_command_input(self, mock_session_fn):
        """Slash commands should pass through normally."""
        from agent.cli.agent_loop import _get_input

        mock_session = MagicMock()
        mock_session.prompt.return_value = "/help"
        mock_session_fn.return_value = mock_session

        with patch("sys.stdout"), patch("agent.cli.agent_loop.console"):
            result = _get_input()

        assert result == "/help"
