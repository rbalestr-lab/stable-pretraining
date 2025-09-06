# Repository Guidelines

## Project Structure & Module Organization
- `stable_pretraining/`: Core library (data, callbacks, optim, losses, utils, manager, module).
- `stable_pretraining/tests/`: Pytest suite (unit/integration). Test discovery is scoped here.
- `examples/`: Runnable scripts and config samples (e.g., `examples/supervised_learning.py`).
- `benchmarks/`: Reproducible method/config baselines.
- `docs/`: Sphinx documentation (`make html`).
- `assets/`, `data/`: Static assets and local data cache (do not commit large files).

## Build, Test, and Development Commands
- Install (editable + dev): `pip install -e .[dev]` (or `uv pip install -e .[dev]`).
- Lint & format (pre-commit): `pre-commit install` then `pre-commit run --all-files`.
- Ruff directly: `ruff .` and `ruff format .`.
- Tests (unit default): `pytest stable_pretraining/ -m unit --cov=stable_pretraining`.
- Full tests: `pytest` (use markers to include/exclude; see below).
- Docs: `cd docs && make html`.
- Example run: `python examples/supervised_learning.py`.

## Coding Style & Naming Conventions
- Python 3.10+, 4‑space indentation, type hints encouraged.
- Docstrings follow Google style (pydocstyle via Ruff). Keep functions/classes documented.
- Naming: modules/files `snake_case.py`; functions/vars `snake_case`; classes `PascalCase`; constants `UPPER_SNAKE`.
- Imports: prefer absolute; avoid unused imports; no prints—use `loguru` or Lightning logging.
- Formatting/linting: enforced by Ruff (see `.pre-commit-config.yaml`, `pyproject.toml`).

## Testing Guidelines
- Framework: Pytest. Markers defined in `pytest.ini`: `unit`, `integration`, `gpu`, `slow`, `download`.
- Naming: files `test_*.py`, classes `Test*`, functions `test_*`.
- Expectations: add unit tests for new code, mark slow/GPU/integration appropriately, keep unit tests fast (<1s) and offline.
- Coverage: run with `--cov=stable_pretraining`; aim to cover new/changed paths.

## Commit & Pull Request Guidelines
- Commits: concise subject; optionally prefix with tag used in history (e.g., `[BUGFIX]`, `[Doc]`, `[Cleaning]`, `[EXAMPLE]`, `[Naming]`).
- PRs: clear description, link issues, include tests and docs updates, keep CI green, add RELEASES.rst entry when user-facing.
- Attach logs or screenshots for training/metrics changes when helpful.

## Security & Configuration Tips
- Do not commit credentials or large datasets. Use `wandb login` / `huggingface-cli login` locally.
- Prefer small, cached datasets for tests; use provided mocks in `stable_pretraining/tests/utils.py`.
- Keep data under `data/` and artifacts under `wandb/` out of commits.

