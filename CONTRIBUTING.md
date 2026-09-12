# Contributing to dscompanion

## Setup

```bash
git clone https://github.com/dsnaveen/dscompanion.git
cd dscompanion
pip install -e ".[dev]"
```

Optional extras for working on specific parts of the project:

```bash
pip install -e ".[app]"    # Streamlit UI
pip install -e ".[api]"    # FastAPI service
pip install -e ".[docs]"   # Sphinx documentation build
```

## Running Tests

```bash
pytest
```

## Formatting and Linting

Both must pass with zero errors before a PR is merged:

```bash
black .
ruff check .
```

## Pull Requests

- Keep PRs focused on one change.
- Add or update tests for any behavior change.
- Add or update docstrings (Google style) for any new or modified public function/class.
- Run `black` and `ruff` locally before opening a PR — CI enforces both.
