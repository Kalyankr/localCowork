import tempfile
import subprocess
import logging
import os
from pathlib import Path

from agent.config import settings

logger = logging.getLogger(__name__)


class Sandbox:
    """
    A sandbox for Python code execution with two modes:

    1. RESTRICTED (default): Docker-isolated, no network, read-only
    2. PERMISSIVE: Direct execution with full file/network access

    Use permissive mode for Cowork-style agentic tasks where the agent
    needs to actually manipulate files, fetch URLs, etc.
    """

    def __init__(self, timeout: int | None = None, permissive: bool = True):
        self.timeout = timeout or settings.sandbox_timeout
        self.permissive = permissive  # Default to permissive for agentic mode
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
        """
        logger.debug(f"Running Python code in permissive mode ({len(code)} chars)")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            script_path = f.name

        try:
            # Run with the user's Python, full access
            env = os.environ.copy()
            cwd = working_dir or os.path.expanduser("~")

            result = subprocess.run(
                ["python", script_path],
                capture_output=True,
                timeout=self.timeout,
                cwd=cwd,
                env=env,
            )

            if result.returncode == 0:
                return {"output": result.stdout.decode()}
            else:
                return {
                    "error": f"Exit code {result.returncode}:\n{result.stderr.decode()}",
                    "output": result.stdout.decode() if result.stdout else None,
                }

        except subprocess.TimeoutExpired:
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
                # Optionally add a strict seccomp profile if available
                # "--security-opt", "seccomp=unconfined",  # Replace with a custom profile for stricter isolation
                "-v",
                f"{tmpdir}:/app:ro",
                "--workdir",
                "/app",
                settings.docker_image,
                "python",
                "script.py",
            ]

            try:
                out = subprocess.check_output(
                    cmd, stderr=subprocess.STDOUT, timeout=self.timeout
                )
                return {"output": out.decode()}

            except subprocess.CalledProcessError as e:
                return {
                    "error": f"Runtime error (exit {e.returncode}):\n{e.output.decode()}"
                }

            except subprocess.TimeoutExpired:
                return {"error": "Execution timed out"}
