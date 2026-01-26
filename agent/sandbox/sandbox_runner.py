import tempfile
import subprocess
import logging
from pathlib import Path

from agent.config import settings

logger = logging.getLogger(__name__)


class Sandbox:
    """
    A secure sandbox that:
    - Creates a temporary directory for each execution
    - Writes the user code into script.py
    - Runs it inside a Python Docker container
    - Mounts ONLY the temp directory
    - Enforces resource limits (memory, CPU, PIDs)
    - Returns stdout or error messages
    """

    def __init__(self, timeout: int | None = None):
        self.timeout = timeout or settings.sandbox_timeout
        self._docker_available: bool | None = None
    
    def _check_docker(self) -> bool:
        """Check if Docker is available."""
        if self._docker_available is None:
            try:
                subprocess.run(
                    ["docker", "info"],
                    capture_output=True,
                    timeout=5
                )
                self._docker_available = True
            except (subprocess.SubprocessError, FileNotFoundError):
                self._docker_available = False
                logger.warning("Docker is not available for sandboxed execution")
        return self._docker_available

    async def run_python(self, code: str) -> dict:
        """
        Execute Python code inside an isolated Docker sandbox.
        Returns a dict with either {"output": "..."} or {"error": "..."}.
        """
        if not self._check_docker():
            return {"error": "Docker is not available. Please install and start Docker."}
        
        logger.debug(f"Running sandboxed Python code ({len(code)} chars)")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            script_path = tmp_path / "script.py"

            # Write the code to the sandbox
            script_path.write_text(code)

            # Docker command with resource limits
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
