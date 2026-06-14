# Contributing to askgraph

Thanks for your interest! Issues and PRs are welcome — especially:

- more first-class languages and deeper tree-sitter relationships (calls, types)
- richer graph algorithms and retrieval-quality improvements
- TUI / MCP UX polish
- benchmarks on more public repos

## Development setup

```bash
git clone https://github.com/bakhliustov/askgraph
cd askgraph
uv sync --group dev
```

This installs the package (editable) plus all extras and dev tools.

## Before opening a PR

The CI runs these on Python 3.11 and 3.12 — please make sure they pass locally:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest -q
```

`ruff format .` will auto-format; `ruff check . --fix` fixes most lint issues.

## Guidelines

- Keep it **local-first**: no feature should require sending code or queries to a
  remote service by default. Network calls (other than a user's own Ollama) should
  be opt-in and clearly gated by `local_only`.
- Match the surrounding style — type hints, short docstrings, `ruff`-clean.
- Add a test for new behavior. Fast/offline unit tests live in `tests/test_units.py`
  and `tests/test_graph_report.py`; the model-downloading integration test lives in
  `tests/test_basic.py`.
- Update `CHANGELOG.md` under an `Unreleased` section for user-facing changes.

## Reporting bugs

Open an issue with the command you ran, the output, your OS/Python version, and
which extras are installed (`git`, `tui`, `mcp`, `tree-sitter-full`).
