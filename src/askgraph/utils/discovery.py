"""File discovery that respects .gitignore and common ignore patterns (local only)."""

from __future__ import annotations

import contextlib
import fnmatch
from pathlib import Path

from askgraph.config import settings
from askgraph.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_IGNORES = [
    ".git",
    ".askgraph",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    ".env",
    "dist",
    "build",
    ".next",
    ".nuxt",
    "target",
    "autogen",  # machine-generated bindings (e.g. ctypes) — huge and low-signal
    "generated",
    "*.pyc",
    "*.pyo",
    ".DS_Store",
    "*.egg-info",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    "assets",  # skip minified JS/CSS in viz/assets etc.
    "*.min.js",
    "*highlight*.js",
    "*d3*.js",
    "*dagre*.js",
]


def discover_files(
    root: Path,
    extensions: list[str] | None = None,
    respect_gitignore: bool = True,
) -> list[Path]:
    """Recursively discover source files under root.

    For MVP we focus on common code extensions. Later we will read .gitignore properly.
    """
    if extensions is None:
        extensions = [
            ".py",
            ".js",
            ".ts",
            ".tsx",
            ".jsx",
            ".go",
            ".rs",
            ".java",
            ".kt",
            ".scala",
            ".md",
            ".txt",
        ]

    root = root.resolve()
    files: list[Path] = []
    skipped_large = 0

    gitignore_path = root / ".gitignore"
    extra_ignores: list[str] = []
    if respect_gitignore and gitignore_path.exists():
        with contextlib.suppress(Exception):
            extra_ignores = [
                line.strip()
                for line in gitignore_path.read_text().splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]

    all_ignores = DEFAULT_IGNORES + extra_ignores

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        # Quick ignore check
        rel = path.relative_to(root)
        rel_str = str(rel)
        ignored = False
        for pattern in all_ignores:
            if fnmatch.fnmatch(rel_str, pattern) or fnmatch.fnmatch(path.name, pattern):
                ignored = True
                break
            # Also check any parent dir segment
            for part in rel.parts:
                if fnmatch.fnmatch(part, pattern):
                    ignored = True
                    break

        if ignored:
            continue

        if not any(rel_str.endswith(ext) or path.suffix == ext for ext in extensions):
            continue

        # Skip oversized files (typically generated/vendored blobs) — they bloat the
        # index and embedding time without adding much for codebase QA.
        if settings.max_file_bytes > 0:
            try:
                if path.stat().st_size > settings.max_file_bytes:
                    skipped_large += 1
                    continue
            except OSError:
                continue

        files.append(path)

    if skipped_large:
        logger.info(
            "Skipped %d oversized file(s) (> %d bytes)", skipped_large, settings.max_file_bytes
        )
    logger.info("Discovered %d files under %s", len(files), root)
    return sorted(files)
