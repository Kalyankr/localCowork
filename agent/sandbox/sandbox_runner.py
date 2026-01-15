import tempfile
import subprocess
from pathlib import Path


class Sandbox:
    """
    A secure sandbox that:
    - Creates a temporary directory for each execution
    - Writes the user code into script.py
    - Runs it inside a Python Docker container
    - Mounts ONLY the temp directory
    - Returns stdout or error messages
    """

    def __init__(self, timeout: int = 20):
        self.timeout = timeout

    async def run_python(self, code: str) -> dict:
        """
        Execute Python code inside an isolated Docker sandbox.
        Returns a dict with either {"output": "..."} or {"error": "..."}.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            script_path = tmp_path / "script.py"

            # Write the code to the sandbox
            script_path.write_text(code)

            # Docker command
            cmd = [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "-v",
                f"{tmpdir}:/app",
                "--workdir",
                "/app",
                "python:3.12-slim",
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
