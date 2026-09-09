# AI Agent Rules: Branch Verification, Quality Standards & Commits

These rules apply unconditionally to all AI agents operating within this workspace. Full detailed guidelines are maintained in [`AGENTS.md`](../../AGENTS.md).

## 1. Mandatory Step 0: Git Branch Check
- **Always run `git branch --show-current` or `git status` first** before modifying or creating any code, tests, documentation, or configuration.
- **Always ask the user** whether to work on the current branch or create a new dedicated branch.
- **Never commit directly to `main`** without explicit user confirmation.
- Standard branch naming: `feat/<name>`, `fix/<name>`, `docs/<name>`, `refactor/<name>`, `test/<name>`.

## 2. High Coding Standards
- **Python 3.12+**: Modern union types (`X | Y`), built-in generics (`list[str]`), no obsolete typing constructs.
- **Strict Typing**: 100% type annotations on all function and method parameters and return values (`disallow_untyped_defs = true` in `pyproject.toml`). Run `uv run mypy src tests`.
- **Ruff Lint & Format**: Enforce 88 character line length, double quotes, clean imports (`uv run ruff check --fix`, `uv run ruff format`).
- **Automated Tests**: Unit tests in `tests/` for all new logic. Run `uv run pytest`. Never make live, unmocked LLM API calls in automated tests—use offline test doubles.
- **Tooling**: Always use `uv` (`uv run`, `uv add`, `uv sync`). Never execute raw `pip install`.
- **Code Integrity**: Preserve existing docstrings, comments, and architectural structure.

## 3. Systematic Commits
- **Incremental & Atomic**: Commit after completing each meaningful, verified milestone. Do not accumulate large, multi-purpose uncommitted diffs.
- **Conventional Commits**: Every commit message must strictly follow `<type>(<optional scope>): <imperative description>`.
- **Quality Gate Before Commit**: Always verify that `ruff check`, `ruff format`, `mypy src tests`, and `pytest` pass before committing.
