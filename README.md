# Statistical Evaluation of LLMs in Data Analysis Tasks

Master's thesis research repository exploring output stochasticity, variance quantification, and minimum sample size estimation ($N^*$) for Large Language Models on tabular data science benchmarks.

* **Author**: Julia Krępińska
* **Institution**: Wrocław University of Science and Technology (Politechnika Wrocławska)
* **Target Defense**: June 2027

## Documentation

* [Thesis PRD & Prospectus](docs/prd.md): Comprehensive thesis proposal, research questions, experimental design, and sprint schedule.
* [Sandbox Documentation](docs/sandbox.md): Execution sandbox architecture, interfaces, and usage guide.
* [Tasks & Masking Documentation](docs/tasks.md): Synthetic missingness mechanisms, NRMSE evaluation, and task definitions.
* [Inference Documentation](docs/inference.md): Unified local and cloud LLM inference engine, hyperparameter grid, and offline test doubles.

## Quickstart

This project uses [`uv`](https://docs.astral.sh/uv/) for Python dependency and virtual environment management:

```bash
# Run test suite
uv run pytest

# Run linting and formatting checks
uv run ruff check
uv run ruff format --check

# Run static type checking
uv run mypy src tests
```

## Contributing & CI

Pull requests targeting `main` automatically run continuous integration checks via GitHub Actions:
* **Tests & Quality**: Runs `pytest`, `ruff` (linter and format check), and `mypy` type checking on Python 3.12.
* **Conventional Commits**: PR titles must follow the [Conventional Commits](https://www.conventionalcommits.org/) format (e.g. `feat: add task runner`, `fix: resolve timeout in sandbox`).

To install pre-commit hooks locally (including commit message checking):
```bash
uv run pre-commit install --hook-type pre-commit --hook-type commit-msg
```
