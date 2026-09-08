"""Tests for the isolated subprocess sandbox and code extractor."""

from masters.sandbox import ExecutionResult, execute_code, extract_python_code


def test_extract_python_code_fenced() -> None:
    raw = "Here is the code:\n```python\nx = 42\nprint(x)\n```\nDone."
    assert extract_python_code(raw) == "x = 42\nprint(x)"


def test_extract_python_code_no_fence() -> None:
    raw = "x = 42\nprint(x)"
    assert extract_python_code(raw) == "x = 42\nprint(x)"


def test_extract_python_code_multiple_blocks() -> None:
    raw = "```python\na = 1\n```\ntext\n```python\nb = 2\n```"
    assert extract_python_code(raw) == "a = 1\n\nb = 2"


def test_execute_code_success() -> None:
    code = "import sys\nsys.stdout.write('test_out')\n"
    result: ExecutionResult = execute_code(code, timeout_sec=5.0)

    assert result.success is True
    assert result.returncode == 0
    assert result.stdout == "test_out"
    assert result.stderr == ""
    assert result.timed_out is False
    assert result.duration_sec >= 0.0


def test_execute_code_syntax_error() -> None:
    code = "def invalid_syntax(\n"
    result = execute_code(code, timeout_sec=5.0)

    assert result.success is False
    assert result.returncode != 0
    assert "SyntaxError" in result.stderr
    assert result.timed_out is False


def test_execute_code_runtime_error() -> None:
    code = "raise ValueError('intentional failure')"
    result = execute_code(code, timeout_sec=5.0)

    assert result.success is False
    assert result.returncode != 0
    assert "ValueError: intentional failure" in result.stderr
    assert result.timed_out is False


def test_execute_code_timeout() -> None:
    code = "import time\ntime.sleep(2.0)\n"
    result = execute_code(code, timeout_sec=0.2)

    assert result.success is False
    assert result.timed_out is True
    assert result.returncode is None
    assert "Execution timed out" in result.stderr
