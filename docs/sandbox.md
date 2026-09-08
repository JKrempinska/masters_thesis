# Execution Sandbox Module (`masters.sandbox`)

The `masters.sandbox` module provides an isolated, subprocess-based execution environment to safely run LLM-generated Python scripts and extract code from Markdown responses.

## Motivation & Architecture

Large Language Models generate arbitrary Python code. Running untrusted code directly in the primary Python runtime introduces several risks:
1. Infinite loops (e.g., iterative convergence failures).
2. Unhandled runtime exceptions and uncaught tracebacks.
3. Uncontrolled memory usage.

This module guarantees process isolation by executing generated code inside a distinct Python subprocess with strict timeout guards and standard stream capturing.

## Public Interface

### `extract_python_code(raw_response: str) -> str`
Extracts Python source code from Markdown code fences (such as `` ```python ... ``` `` or generic `` ``` ... ``` ``).
* If multiple blocks are detected, they are concatenated in sequence.
* If no markdown fence is found, returns the trimmed raw response.

### `execute_code(code: str, timeout_sec: float = 15.0, working_dir: Path | None = None) -> ExecutionResult`
Executes Python source code in a dedicated temporary directory using `sys.executable`.

#### Parameters
* `code` (`str`): The Python source string to execute.
* `timeout_sec` (`float`, default `15.0`): Maximum permitted execution time before process termination.
* `working_dir` (`Path | None`, optional): Directory context for execution. A temporary directory is used if omitted.

#### Returns
An `ExecutionResult` frozen dataclass with the following attributes:
* `success` (`bool`): `True` if the process terminated with exit code `0`, `False` otherwise.
* `stdout` (`str`): Captured standard output stream.
* `stderr` (`str`): Captured standard error stream and tracebacks.
* `returncode` (`int | None`): Exit status code (`None` if the process timed out).
* `duration_sec` (`float`): Execution duration in seconds.
* `timed_out` (`bool`): `True` if execution was aborted due to exceeding `timeout_sec`.

## Example Usage

````python
from masters.sandbox import execute_code, extract_python_code

raw_llm_response = """
Here is the solution:
```python
import pandas as pd

df = pd.DataFrame({"a": [1, 2, None]})
print(df.dropna().shape)
```
"""

code = extract_python_code(raw_llm_response)
result = execute_code(code, timeout_sec=5.0)

if result.success:
    print("Output:", result.stdout)
else:
    print("Failed with error:", result.stderr)
````
