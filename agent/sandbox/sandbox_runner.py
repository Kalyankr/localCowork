import asyncio
import logging
import os
import subprocess
import tempfile
from pathlib import Path

from agent.config import settings

logger = logging.getLogger(__name__)


class Sandbox:
    """
    A sandbox for Python code execution with two modes:

    1. RESTRICTED (default): Docker-isolated, no network, read-only
    2. PERMISSIVE: Direct execution with full file/network access

    SECURITY NOTE: Permissive mode grants full system access to AI-generated code.
    Only use permissive=True when you trust the environment and need file/network access.
    """

    def __init__(self, timeout: int | None = None, permissive: bool = False):
        self.timeout = timeout or settings.sandbox_timeout
        self.permissive = permissive  # Default to RESTRICTED for security
        self._docker_available: bool | None = None

    def _check_docker(self) -> bool:
        """Check if Docker is available."""
        if self._docker_available is None:
            try:
                subprocess.run(["docker", "info"], capture_output=True, timeout=5)
                self._docker_available = True
            except (subprocess.SubprocessError, FileNotFoundError):
                self._docker_available = False
                logger.warning("Docker is not available for sandboxed execution")
        return self._docker_available

    async def run_python(self, code: str, working_dir: str = None) -> dict:
        """
        Execute Python code.

        In permissive mode: runs directly with full access (like Cowork)
        In restricted mode: runs in Docker sandbox
        """
        if self.permissive:
            return await self._run_permissive(code, working_dir)
        else:
            return await self._run_docker(code)

    async def _run_permissive(self, code: str, working_dir: str = None) -> dict:
        """
        Run Python code directly with full system access.
        This is the Cowork-style mode for agentic tasks.

        SECURITY WARNING: This grants full access to AI-generated code.
        """
        logger.debug(f"Running Python code in permissive mode ({len(code)} chars)")

        # Create temp file with secure permissions
        fd, script_path = tempfile.mkstemp(suffix=".py", text=True)
        try:
            os.chmod(script_path, 0o600)  # Only owner can read/write
            with os.fdopen(fd, "w") as f:
                f.write(code)

            # Run with the user's Python, full access (async)
            cwd = working_dir or os.path.expanduser("~")

            try:
                proc = await asyncio.create_subprocess_exec(
                    "python",
                    script_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=cwd,
                )

                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=self.timeout
                )

                if proc.returncode == 0:
                    return {"output": stdout.decode()}
                else:
                    return {
                        "error": f"Exit code {proc.returncode}:\n{stderr.decode()}",
                        "output": stdout.decode() if stdout else None,
                    }

            except TimeoutError:
                proc.kill()
                return {"error": f"Execution timed out after {self.timeout}s"}
            except Exception as e:
                return {"error": str(e)}
        finally:
            os.unlink(script_path)

    async def _run_docker(self, code: str) -> dict:
        """
        Execute Python code inside an isolated Docker sandbox.
        Returns a dict with either {"output": "..."} or {"error": "..."}.
        """
        if not self._check_docker():
            return {
                "error": "Docker is not available. Please install and start Docker."
            }

        logger.debug(f"Running sandboxed Python code ({len(code)} chars)")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            script_path = tmp_path / "script.py"

            # Write the code to the sandbox
            script_path.write_text(code)

            # Docker command with enhanced security
            cmd = [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--memory",
                settings.sandbox_memory_limit,
                "--cpus",
                settings.sandbox_cpu_limit,
                "--pids-limit",
                str(settings.sandbox_pids_limit),
                "--user",
                settings.sandbox_user_id,
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--read-only",
                "--tmpfs",
                "/tmp:size=10m,exec,nosuid,nodev",
                "-v",
                f"{tmpdir}:/app:ro",
                "--workdir",
                "/app",
                settings.docker_image,
                "python",
                "script.py",
            ]

            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )

                stdout, _ = await asyncio.wait_for(
                    proc.communicate(), timeout=self.timeout
                )

                if proc.returncode == 0:
                    return {"output": stdout.decode()}
                else:
                    return {
                        "error": f"Runtime error (exit {proc.returncode}):\n{stdout.decode()}"
                    }

            except TimeoutError:
                proc.kill()
                return {"error": "Execution timed out"}
