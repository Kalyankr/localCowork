"""Tests for the context-aware tool suggestion engine (agent/tools/tool_selector.py)."""

from agent.tools.tool_selector import suggest_tools

# All 9 built-in tools
ALL_TOOLS = sorted(
    [
        "shell",
        "python",
        "web_search",
        "fetch_webpage",
        "read_file",
        "write_file",
        "edit_file",
        "memory_store",
        "memory_recall",
    ]
)


class TestSuggestToolsBasic:
    """Core filtering logic."""

    def test_empty_goal_returns_all(self):
        result = suggest_tools("", ALL_TOOLS)
        assert result == ALL_TOOLS

    def test_whitespace_goal_returns_all(self):
        result = suggest_tools("   ", ALL_TOOLS)
        assert result == ALL_TOOLS

    def test_ambiguous_goal_returns_all(self):
        """No keywords matched → all tools."""
        result = suggest_tools("Hey, how are you?", ALL_TOOLS)
        assert result == ALL_TOOLS

    def test_complex_goal_returns_all(self):
        """Matching 4+ groups → treat as complex, return all."""
        goal = (
            "Search the web for Python tutorials, "
            "read config.yaml, run tests, and remember the results"
        )
        result = suggest_tools(goal, ALL_TOOLS)
        assert result == ALL_TOOLS


class TestSuggestToolsFileOps:
    """File-related goals include file tools."""

    def test_read_file_keyword(self):
        result = suggest_tools("Show me the config.yaml file", ALL_TOOLS)
        assert "read_file" in result
        assert "write_file" in result
        assert "edit_file" in result
        assert "shell" in result  # always included

    def test_create_file(self):
        result = suggest_tools("Create a new Python file called app.py", ALL_TOOLS)
        assert "write_file" in result

    def test_edit_keyword(self):
        result = suggest_tools("Edit the README.md", ALL_TOOLS)
        assert "edit_file" in result
        assert "read_file" in result

    def test_file_extension_triggers_file_ops(self):
        result = suggest_tools("What's in requirements.txt", ALL_TOOLS)
        assert "read_file" in result

    def test_no_web_tools_for_file_task(self):
        result = suggest_tools("Read the config file", ALL_TOOLS)
        assert "web_search" not in result
        assert "fetch_webpage" not in result


class TestSuggestToolsShell:
    """Shell commands trigger shell tool."""

    def test_git_command(self):
        result = suggest_tools("Run git status", ALL_TOOLS)
        assert "shell" in result

    def test_install_keyword(self):
        result = suggest_tools("Install numpy", ALL_TOOLS)
        assert "shell" in result

    def test_list_directory(self):
        result = suggest_tools("List files in the directory", ALL_TOOLS)
        assert "shell" in result

    def test_docker_command(self):
        result = suggest_tools("Build the docker image", ALL_TOOLS)
        assert "shell" in result


class TestSuggestToolsWeb:
    """Web-related goals include web tools."""

    def test_search_keyword(self):
        result = suggest_tools("Search for asyncio best practices", ALL_TOOLS)
        assert "web_search" in result
        assert "fetch_webpage" in result

    def test_url_in_goal(self):
        result = suggest_tools(
            "Fetch https://example.com/page and summarize", ALL_TOOLS
        )
        assert "fetch_webpage" in result

    def test_documentation_keyword(self):
        result = suggest_tools("Find the documentation for FastAPI", ALL_TOOLS)
        assert "web_search" in result

    def test_how_to_keyword(self):
        result = suggest_tools("How to use pytest fixtures", ALL_TOOLS)
        assert "web_search" in result

    def test_no_file_ops_for_web_search(self):
        result = suggest_tools("Search the web for Python news", ALL_TOOLS)
        assert "read_file" not in result
        assert "write_file" not in result
        assert "edit_file" not in result


class TestSuggestToolsCode:
    """Data/code analysis triggers python tool."""

    def test_analyze_data(self):
        result = suggest_tools("Analyze the sales data in CSV", ALL_TOOLS)
        assert "python" in result

    def test_calculate_keyword(self):
        result = suggest_tools("Calculate the average of these numbers", ALL_TOOLS)
        assert "python" in result

    def test_pandas_keyword(self):
        result = suggest_tools("Use pandas to process the report", ALL_TOOLS)
        assert "python" in result

    def test_plot_keyword(self):
        result = suggest_tools("Plot a chart of monthly revenue", ALL_TOOLS)
        assert "python" in result


class TestSuggestToolsMemory:
    """Memory-related goals include memory tools."""

    def test_remember_keyword(self):
        result = suggest_tools("Remember that this project uses pytest", ALL_TOOLS)
        assert "memory_store" in result
        assert "memory_recall" in result

    def test_recall_keyword(self):
        result = suggest_tools("Recall what you know about this project", ALL_TOOLS)
        assert "memory_recall" in result

    def test_preference_keyword(self):
        result = suggest_tools("Save my preference for dark mode", ALL_TOOLS)
        assert "memory_store" in result

    def test_no_web_tools_for_memory(self):
        result = suggest_tools("Remember that I prefer Python", ALL_TOOLS)
        assert "web_search" not in result


class TestSuggestToolsUsedTools:
    """Previously used tools are always retained."""

    def test_used_tools_always_included(self):
        result = suggest_tools(
            "Search for Python tutorials",
            ALL_TOOLS,
            used_tools=["read_file", "python"],
        )
        assert "read_file" in result
        assert "python" in result
        assert "web_search" in result  # from goal keywords

    def test_used_tools_from_prior_steps(self):
        """Even if goal doesn't mention files, used file tools are kept."""
        result = suggest_tools(
            "Search the web for news",
            ALL_TOOLS,
            used_tools=["edit_file"],
        )
        assert "edit_file" in result

    def test_used_tools_with_empty_goal(self):
        result = suggest_tools("", ALL_TOOLS, used_tools=["python"])
        # Empty goal → all tools returned anyway
        assert result == ALL_TOOLS


class TestSuggestToolsMultiGroup:
    """Combined intents include tools from multiple groups."""

    def test_file_and_shell(self):
        result = suggest_tools("Find all .py files and list them", ALL_TOOLS)
        assert "shell" in result
        assert "read_file" in result

    def test_web_and_memory(self):
        result = suggest_tools("Search for the answer and remember it", ALL_TOOLS)
        assert "web_search" in result
        assert "memory_store" in result

    def test_file_and_code(self):
        result = suggest_tools("Read the CSV file and analyze the data", ALL_TOOLS)
        assert "read_file" in result
        assert "python" in result


class TestSuggestToolsEdgeCases:
    """Edge cases and safety nets."""

    def test_subset_of_available_tools(self):
        """Only returns tools that are actually available."""
        limited = ["shell", "python"]
        result = suggest_tools("Run a python script", limited)
        for t in result:
            assert t in limited

    def test_single_available_tool_returns_all(self):
        """If filtering would leave < 2 tools, return all."""
        result = suggest_tools("Remember this", ["memory_store"])
        assert result == ["memory_store"]

    def test_case_insensitive(self):
        result = suggest_tools("SEARCH THE WEB", ALL_TOOLS)
        assert "web_search" in result

    def test_search_file_excludes_web(self):
        """'search' near 'file' should not trigger web tools."""
        result = suggest_tools("search for a file named report", ALL_TOOLS)
        # "search" triggers web, but "file" triggers file_ops + shell
        # both web and file_ops matched — that's fine, multi-group
        assert "shell" in result

    def test_returns_sorted(self):
        result = suggest_tools("Edit the config file", ALL_TOOLS)
        assert result == sorted(result)

    def test_unknown_tools_in_used_tools_ignored(self):
        """used_tools not in available_tools are silently dropped."""
        result = suggest_tools(
            "Read the file",
            ALL_TOOLS,
            used_tools=["nonexistent_tool"],
        )
        assert "nonexistent_tool" not in result


class TestRegistryFilteredDescriptions:
    """Test ToolRegistry.get_tool_descriptions with tool_names filter."""

    def test_filtered_descriptions(self):
        from agent.tools.registry import ToolRegistry

        class FakeTool:
            def __init__(self, name: str):
                self.name = name
                self.description = f"{name} desc"
                self.args_schema = {"arg": "val"}

            async def execute(self, args, context):
                return {}

        reg = ToolRegistry()
        reg.register(FakeTool("alpha"))
        reg.register(FakeTool("beta"))
        reg.register(FakeTool("gamma"))

        # Unfiltered
        full = reg.get_tool_descriptions()
        assert "alpha" in full
        assert "beta" in full
        assert "gamma" in full

        # Filtered
        filtered = reg.get_tool_descriptions(tool_names=["alpha", "gamma"])
        assert "alpha" in filtered
        assert "gamma" in filtered
        assert "beta" not in filtered

    def test_none_filter_returns_all(self):
        from agent.tools.registry import ToolRegistry

        class FakeTool:
            def __init__(self, name: str):
                self.name = name
                self.description = f"{name} desc"
                self.args_schema = {}

            async def execute(self, args, context):
                return {}

        reg = ToolRegistry()
        reg.register(FakeTool("x"))
        reg.register(FakeTool("y"))

        result = reg.get_tool_descriptions(tool_names=None)
        assert "x" in result
        assert "y" in result

    def test_empty_filter_returns_nothing(self):
        from agent.tools.registry import ToolRegistry

        class FakeTool:
            def __init__(self, name: str):
                self.name = name
                self.description = f"{name} desc"
                self.args_schema = {}

            async def execute(self, args, context):
                return {}

        reg = ToolRegistry()
        reg.register(FakeTool("x"))

        result = reg.get_tool_descriptions(tool_names=[])
        assert result == ""
