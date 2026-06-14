# Changelog

All notable changes to askgraph will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-06-12

### Added
- Full per-symbol git blame history (`git_history` with up to 5 recent commits per symbol: commit, author, date, message).
- Deeper graph analysis:
  - `detect_communities` using networkx greedy modularity (with fallback).
  - `compute_blast_radius` for impact analysis (nodes/edges within N hops of a symbol/file).
- Lightweight eval harness: `askgraph eval` command runs sample queries on a codebase and reports retrieval stats, symbol/provenance coverage, and estimated token savings. Easy to run on public repos for benchmarks.
- Full MCP server support (`[mcp]` extra): Native tools for agents (`index_codebase`, `ask`, `get_god_nodes`, `get_communities`, `compute_blast_radius`, `export_agent_context`). Run with `askgraph mcp .`.
- Rich agent exports via MCP `export_agent_context` (structured JSON with hybrid sources, subgraph, provenance, token estimate, suggested prompt template).
- Polished TUI (`[tui]` extra): Improved layout, CSS, provenance visibility in sidebar and sources, better god-node display with git info, cleaner command handling.
- Release polish:
  - Basic tests (`tests/test_basic.py`).
  - GitHub Actions CI (lint, typecheck, tests on py311/312).
  - ASCII logo and updated branding in README.
  - Comprehensive launch announcement draft in README.
  - Version bumped to 0.2.0.
- Expanded public repo demo instructions in README (FastAPI, Pydantic, Flask, Starlette, Requests, Express) with real example outputs from indexing `psf/requests`.
- Benchmarks section in README with realistic numbers from public repo runs.
- All new graph/provenance features integrated into:
  - Synthesis prompts and answers (LLM references history when available).
  - `GRAPH_REPORT.md` (communities, blast radius, enhanced god nodes with git).
  - TUI (sidebar/sources show provenance).
  - MCP tools and agent packs.
  - `export` command output.

### Changed
- Indexer now collects richer per-symbol git data across the full line range (not just last-touch).
- Report generator and MCP expose deeper structural insights.
- README restructured for launch: Agent/MCP section upfront, public demos with copy-paste, status reflecting 0.2.0 features.

### Fixed
- **Packaging:** added a `[build-system]` (hatchling) so the project actually builds, installs, and exposes the `askgraph` console script. Previously `pip`/`uv` install produced no entry point.
- **MCP server:** `get_communities` and `compute_blast_radius` referenced an unimported `nx` and crashed on every call; they now reuse the shared graph builder.
- **Indexer:** file-level git enrichment referenced an undefined `target_dir` and silently never ran.
- **Git provenance:** "last touched" now reports the *most recent* commit in a symbol's line range (previously it reported the first line's commit, regardless of date).
- **Performance:** git blame is now run once per file instead of once per symbol, and is gated behind the `use_git_blame` setting (it was previously always-on whenever GitPython was installed).
- **Structural graph:** non-Python files (Go/Rust/Markdown/…) are no longer mis-parsed with the Python grammar, which had been polluting the graph with bogus symbols.
- **Config:** the embedding model is now a single source of truth (`ASKGRAPH_EMBEDDING_MODEL`) used by both the indexer and retriever, instead of a dead setting plus two hardcoded values.
- MCP server now honors the codebase path it was launched with as the default for tool calls.
- Whole codebase now passes `ruff check`, `ruff format --check`, and strict `mypy`.

## [0.1.0] - Initial (pre-0.2)
- Core local indexing with tree-sitter structural parsing + fastembed + Chroma.
- Hybrid retrieval (semantic + graph neighborhoods).
- Ollama synthesis with citations.
- Structural graph.json, GRAPH_REPORT.md, self-contained graph.html.
- CLI (index, ask, status, report, export, tui, mcp).
- Git enrichment (file + per-symbol last touch).
- TUI chat with god nodes sidebar.
- Provenance in answers and reports.
- Public repo demo focus (not personal projects).

[0.2.0]: https://github.com/bakhliustov/askgraph/compare/v0.1.0...v0.2.0
