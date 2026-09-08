"""Subprocess-based sandbox for isolated, safe execution of Python code."""

import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExecutionResult:
    """Outcome of sandboxed script execution."""

    success: bool
    stdout: str
    stderr: str
    returncode: int | None
    duration_sec: float
    timed_out: bool = False


def execute_code(
    code: str,
    timeout_sec: float = 15.0,
    working_dir: Path | None = None,
) -> ExecutionResult:
    """Execute Python code in an isolated subprocess with a strict timeout limit.

    Args:
        code: The Python source code string to execute.
        timeout_sec: Maximum permitted runtime before process termination.
        working_dir: Directory where execution takes place. Uses a temporary
            directory if omitted.

    Returns:
        ExecutionResult containing exit status, standard streams, and duration.
    """
    start_time = time.perf_counter()

    with tempfile.TemporaryDirectory() as temp_dir:
        exec_dir = working_dir if working_dir is not None else Path(temp_dir)
        script_path = exec_dir / "_sandbox_script.py"
        script_path.write_text(code, encoding="utf-8")

        try:
            completed = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(exec_dir),
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
            duration = time.perf_counter() - start_time
            return ExecutionResult(
                success=completed.returncode == 0,
                stdout=completed.stdout,
                stderr=completed.stderr,
                returncode=completed.returncode,
                duration_sec=duration,
                timed_out=False,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.perf_counter() - start_time
            stdout = exc.stdout if isinstance(exc.stdout, str) else ""
            stderr = exc.stderr if isinstance(exc.stderr, str) else ""
            return ExecutionResult(
                success=False,
                stdout=stdout,
                stderr=stderr + f"\nExecution timed out after {timeout_sec}s.",
                returncode=None,
                duration_sec=duration,
                timed_out=True,
            )
        finally:
            if script_path.exists():
                script_path.unlink()
