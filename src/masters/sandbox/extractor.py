"""Utilities for extracting executable Python code blocks from LLM responses."""

import re

_CODE_BLOCK_PATTERN = re.compile(
    r"```(?:python)?\s*\n(.*?)\n```",
    re.DOTALL | re.IGNORECASE,
)


def extract_python_code(raw_response: str) -> str:
    """Extract Python source code from a Markdown formatted LLM output.

    If multiple markdown blocks are found, concatenates them. If no markdown
    code fence is present, returns the stripped raw response.
    """
    matches = _CODE_BLOCK_PATTERN.findall(raw_response)
    if matches:
        return "\n\n".join(match.strip() for match in matches)
    return raw_response.strip()
