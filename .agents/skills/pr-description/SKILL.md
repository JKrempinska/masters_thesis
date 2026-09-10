---
name: pr-description
description: >-
  Draft, format, and generate Conventional Commit Pull Request (PR) titles and structured descriptions
  featuring an overview, most important changes, and verification checklist. Use whenever the user asks
  to create a PR, generate a PR name/description, review branch diffs for GitHub, or run `gh pr create`.
---

# Pull Request Title & Description Generator

This skill guides the agent in analyzing branch git diffs and commit history to produce clean, high-standard Pull Request (PR) titles and structured descriptions adhering to the repository's guidelines in `AGENTS.md`.

---

## Workflow Steps

### Step 1: Branch & Diff Analysis
Before writing the PR title or description, inspect the changes between the active branch and the base branch (default: `main`):

1. **Verify Current Branch**:
   ```bash
   git branch --show-current
   ```
2. **Review Commit History on Branch**:
   ```bash
   git log main..HEAD --oneline
   ```
3. **Inspect File Changes & Statistics**:
   ```bash
   git diff main...HEAD --stat
   ```
4. **Examine Key Diffs**:
   Inspect significant code modifications to understand architectural decisions, interfaces added, bugfixes, or refactors.

---

### Step 2: Formulate Conventional Commit PR Title

Format the PR title strictly according to the **Conventional Commits** specification defined in `AGENTS.md`:

```text
<type>(<scope>): <short description in imperative mood>
```

#### Permitted Types:
* `feat`: A new feature or capability (e.g. inference engine, sandbox runner, masking strategy).
* `fix`: A bug fix or error correction.
* `docs`: Documentation updates, PRD updates, ADRs, or README changes.
* `refactor`: Structural code improvements without behavioral changes.
* `test`: Adding missing tests or improving existing test infrastructure.
* `perf`: Performance optimizations (e.g. caching, vectorized computations).
* `ci`: Changes to CI configuration, GitHub Actions, or pre-commit hooks.
* `chore`: Dependency bumps, formatting configs, or routine maintenance.

#### Scope Guidelines:
* Use the primary module or feature name: `(sandbox)`, `(inference)`, `(tasks)`, `(stats)`, `(adr)`, `(ci)`, `(skills)`.
* Keep the summary concise (under 72 characters), lowercase, and in the imperative mood (*"implement"*, not *"implemented"* or *"implements"*). No period at the end.

---

### Step 3: Draft Structured PR Description

Use the template defined in [template.md](./resources/template.md):

```markdown
## Overview
<!-- 1-3 sentences explaining the motivation, context, and what this pull request accomplishes. Reference relevant PRD sprint, ADR, or issue if applicable. -->

## Key Changes
<!-- Bulleted list highlighting the most important technical and architectural changes. Focus on non-obvious design decisions, new interfaces, or algorithmic changes rather than trivial line edits. -->
- **<Area / Component>**: <Key modification and rationale>
- **<Area / Component>**: <Key modification and rationale>

## Verification
<!-- Verification status of repository quality gates. -->
- [x] Linting passed (`uv run ruff check`)
- [x] Code formatting passed (`uv run ruff format --check`)
- [x] Static type checking passed (`uv run mypy src tests`)
- [x] Unit test suite passed (`uv run pytest`)
```

#### Writing Rules:
1. **Focus on Substance**: Highlight *why* changes were made and *how* components interact, not just which files were touched.
2. **Clarity & Brevity**: Keep the overview punchy and the key changes bulleted with bold component prefixes.
3. **Integrity & Honesty**: Only check off verification items that have been executed and verified green.

---

### Step 4: Dual Delivery to User

Always provide the user with two ways to submit their PR:

1. **Markdown Box for Web UI**:
   Present the formatted Title and Body in fenced code blocks so the user can easily copy and paste into github.com:

   ```markdown
   ### Suggested PR Title
   `feat(scope): imperative summary`

   ### Suggested PR Description
   ```markdown
   ## Overview
   ...

   ## Key Changes
   ...

   ## Verification
   ...
   ```
   ```

2. **GitHub CLI Command**:
   Provide an executable `gh pr create` command:
   ```bash
   gh pr create --title "<PR_TITLE>" --body "<ESCAPED_PR_BODY>"
   ```
   If the branch has not been pushed to remote yet, mention the push command first:
   ```bash
   git push -u origin <current-branch>
   ```
