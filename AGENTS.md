# AI Agent Guidelines

This repository contains research and codebase for the Master's thesis **"Statistical Evaluation of LLMs in Data Analysis Tasks"** (Julia Krępińska, Wrocław University of Science and Technology).

All AI agents (Antigravity, Cursor, Copilot, etc.) working in this repository must strictly adhere to the standards and workflows defined in this document.

---

## 1. Mandatory Step 0: Branch Verification & Confirmation

**Before performing any modifications** to code, documentation, tests, or configurations:

1. **Check Current Branch**:
   Run `git branch --show-current` or `git status` to determine the active branch.
2. **Prompt the User**:
   Inform the user of the current branch and explicitly ask whether to:
   - Continue working on the current branch, OR
   - Create a new feature/fix/docs branch.
3. **Branch Naming Convention**:
   If creating a new branch, branch off the latest `main` (or designated base) and follow this naming schema:
   - `feat/<short-description>` (for new features, models, runners)
   - `fix/<short-description>` (for bugfixes)
   - `docs/<short-description>` (for documentation, PRD updates)
   - `refactor/<short-description>` (for structural code improvements)
   - `test/<short-description>` (for testing infrastructure)
4. **Protected Main**:
   **Never** make direct commits to `main` without explicit, unambiguous user confirmation.

---

## 2. High Coding Standards

### 2.1. Python 3.12+ Modern Practices
- Target Python version is **3.12+**.
- Utilize modern syntax and typing:
  - Use `|` for union types (e.g., `str | None`, `int | float`), not `Optional[...]` or `Union[...]`.
  - Use built-in generics (e.g., `list[str]`, `dict[str, Any]`, `tuple[int, ...]`), not `typing.List` or `typing.Dict`.
  - Use Python 3.12 type parameter syntax where applicable (e.g., `def func[T](val: T) -> T:`).

### 2.2. Strict Static Typing (Mypy)
- **100% Type Annotation**: All function signatures, method arguments, return types, and class attributes must be fully typed.
- `disallow_untyped_defs = true` is enabled in `pyproject.toml`.
- Avoid `Any` whenever possible. Use `TypeVar`, `Protocol`, `TypedDict`, `Literal`, or concrete types.
- Always verify types using:
  ```bash
  uv run mypy src tests
  ```

### 2.3. Code Formatting & Linting (Ruff)
- Formatter & linter is **Ruff** (configured in `pyproject.toml`):
  - Line length: 88 characters.
  - Quote style: Double quotes (`"`).
  - Selected rules: `E`, `W` (pycodestyle), `F` (pyflakes), `I` (isort / import ordering), `B` (flake8-bugbear), `UP` (pyupgrade), `C4` (flake8-comprehensions).
- Always ensure code is cleanly formatted and lint-free:
  ```bash
  uv run ruff check --fix
  uv run ruff format
  ```

### 2.4. Unit Testing & Test Doubles
- Tests reside in `tests/` and should mirror the structure of `src/masters/`.
- Every new module, feature, algorithm, or bugfix must include corresponding automated tests.
- **Fast & Deterministic**: Unit tests must run quickly without flaky behavior.
- **No Uncontrolled Live LLM Calls in Tests**:
  - LLM inference must never make unmocked, paid, or network-dependent API calls during unit tests.
  - Use offline test doubles, fixtures, or stub responses (see `docs/inference.md`).
- All tests must pass before completing work:
  ```bash
  uv run pytest
  ```

### 2.5. Architecture & Code Integrity
- Maintain separation of concerns:
  - `masters.sandbox`: Execution sandbox architecture and isolation.
  - `masters.tasks`: Synthetic missingness, tasks, evaluation metrics (NRMSE).
  - `masters.inference`: Local/cloud inference engine, parameter grid.
- Provide descriptive docstrings for public classes, functions, and modules.
- **Preserve Documentation Integrity**: Do not delete existing comments, docstrings, or documentation unless explicitly asked or replacing deprecated code.

### 2.6. Package & Dependency Management
- Exclusively use [`uv`](https://docs.astral.sh/uv/):
  - Run commands: `uv run <command>`
  - Add dependencies: `uv add <package>` or `uv add --dev <package>`
  - Sync environment: `uv sync`
- Do not run raw `pip install` or modify virtual environments directly.

---

## 3. Systematic Commits & Git Hygiene

### 3.1. Commit Frequency
- **Commit incrementally**: Do NOT make one giant commit containing all changes at the end of a session.
- Make atomic, systematic commits after completing each **meaningful, self-contained milestone** (e.g., after defining interfaces, after implementing logic, after adding test coverage, or after updating documentation).

### 3.2. Conventional Commits Standard
- All commit messages must strictly follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:
  ```
  <type>(<optional scope>): <description in imperative mood>

  [optional body]
  ```
- Permitted types:
  - `feat`: A new feature or capability
  - `fix`: A bug fix
  - `docs`: Documentation changes only
  - `refactor`: Code change that neither fixes a bug nor adds a feature
  - `test`: Adding missing tests or correcting existing tests
  - `perf`: A code change that improves performance
  - `ci`: Changes to CI configuration and scripts (e.g. GitHub Actions, pre-commit)
  - `chore`: Maintenance tasks, dependency bumps
- Examples:
  - `feat(inference): implement exponential backoff for api retries`
  - `test(sandbox): add unit tests for docker timeout handling`
  - `docs(prd): update sample size estimation section`
  - `fix(tasks): correct NRMSE normalization formula for constant columns`

### 3.3. Pre-Commit Quality Gate
Before executing any git commit, run the full verification suite:
```bash
# 1. Lint & fix autofixable issues
uv run ruff check --fix

# 2. Format code
uv run ruff format

# 3. Static type check
uv run mypy src tests

# 4. Run test suite
uv run pytest
```
All checks must be green before staging and committing.

---

## 4. AI Agent Operational Workflow Checklist

When assigned a task, follow this exact sequence:

- [ ] **Step 1: Verify Git Branch**: Run `git branch --show-current`. Confirm with the user whether to proceed on the current branch or create a new branch.
- [ ] **Step 2: Understand Context**: Inspect relevant files, documentation in `docs/`, and `pyproject.toml`.
- [ ] **Step 3: Implement Incrementally**: Make targeted, minimal, high-quality edits adhering to Python 3.12+ standards and strict typing.
- [ ] **Step 4: Add / Update Tests**: Add tests covering new code and edge cases.
- [ ] **Step 5: Run Verification Suite**: Execute `ruff check`, `ruff format`, `mypy`, and `pytest`.
- [ ] **Step 6: Systematic Commit**: Commit meaningful units of work with Conventional Commit messages.
- [ ] **Step 7: Summarize**: Present a clear, concise walkthrough of the changes to the user.
