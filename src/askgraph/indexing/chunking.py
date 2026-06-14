"""Code-aware chunking with strong structural support via tree-sitter.

For Python (and soon other languages) we prefer tree-sitter extracted symbols
(functions, classes) as the primary chunk units. This is the key quality jump
inspired by Graphify: chunks align with real code architecture instead of arbitrary text splits.

Fallback to high-quality recursive chunking (inspired by finrag-eval) for
non-Python files, glue code, or oversized symbols.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from askgraph.config import settings
from askgraph.indexing.parsers import parse_file
from askgraph.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CodeChunk:
    """A chunk of code or docs with rich metadata for retrieval + graph linking."""

    chunk_id: str
    text: str
    file_path: str
    start_line: int | None = None
    end_line: int | None = None
    symbol: str | None = None  # function/class name when we can extract it
    language: str | None = None
    metadata: dict[str, Any] | None = None

    def to_metadata(self) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "file_path": self.file_path,
        }
        if self.start_line is not None:
            meta["start_line"] = self.start_line
        if self.end_line is not None:
            meta["end_line"] = self.end_line
        if self.symbol:
            meta["symbol"] = self.symbol
        if self.language:
            meta["language"] = self.language
        if self.metadata:
            meta.update(self.metadata)
        return meta


def _recursive_chunk(
    text: str,
    file_path: str,
    language: str | None,
    chunk_size: int,
    overlap: int,
    base_id: str,
    metadata_extra: dict[str, Any] | None = None,
) -> list[CodeChunk]:
    """Internal recursive text chunker (used for fallback and oversized symbols)."""
    if len(text) <= chunk_size:
        line_count = text.count("\n") + 1
        return [
            CodeChunk(
                chunk_id=base_id,
                text=text,
                file_path=file_path,
                start_line=1,
                end_line=line_count,
                language=language,
                metadata={"strategy": "recursive_fallback", **(metadata_extra or {})},
            )
        ]

    separators = ["\n\nclass ", "\n\ndef ", "\n\nasync def ", "\n\n", "\n", ". ", " "]
    chunks: list[CodeChunk] = []
    idx = 0
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        piece = text[start:end]

        for sep in separators:
            last = piece.rfind(sep)
            if last > chunk_size // 3:
                end = start + last
                piece = text[start:end]
                break

        chunk = CodeChunk(
            chunk_id=f"{base_id}:{idx}",
            text=piece.strip(),
            file_path=file_path,
            start_line=text[:start].count("\n") + 1,
            end_line=text[:end].count("\n") + 1,
            language=language,
            metadata={"strategy": "recursive_code", **(metadata_extra or {})},
        )
        chunks.append(chunk)
        idx += 1
        start = max(end - overlap, end)
        if start >= len(text):
            break

    return chunks


def chunk_with_tree_sitter(
    text: str, file_path: str, language: str | None = None
) -> list[CodeChunk]:
    """Primary structural chunker using tree-sitter for supported languages.

    Extracts functions/classes as first-class chunks with exact source ranges.
    Large symbols are recursively sub-chunked. Falls back gracefully.
    """
    chunk_size = settings.chunk_size
    overlap = settings.chunk_overlap
    lang = language or "python"

    try:
        parsed = parse_file(text, file_path)
    except Exception as exc:
        logger.warning(
            "tree-sitter parse failed for %s (%s) — falling back to text chunking", file_path, exc
        )
        return chunk_text(text, file_path, language=lang)

    if not parsed.get("symbols"):
        return _recursive_chunk(text, file_path, lang, chunk_size, overlap, f"{file_path}:module")

    source_lines = text.splitlines(keepends=True)
    symbols = sorted(parsed["symbols"], key=lambda s: s.get("start_line", 0))

    chunks: list[CodeChunk] = []
    prev_end_line = 0

    for sym in symbols:
        sym_start_line = sym["start_line"] - 1
        sym_end_line = sym["end_line"]

        if sym_start_line > prev_end_line:
            glue = "".join(source_lines[prev_end_line:sym_start_line]).strip()
            if glue and len(glue) > 20:
                glue_chunks = _recursive_chunk(
                    glue,
                    file_path,
                    lang,
                    chunk_size,
                    overlap,
                    f"{file_path}:glue:{prev_end_line}",
                    {"strategy": "structural_glue"},
                )
                chunks.extend(glue_chunks)

        sym_text = "".join(source_lines[sym_start_line:sym_end_line]).rstrip("\n")
        sym_name = sym["name"]
        sym_type = sym["type"]

        base_meta = {"strategy": "tree_sitter_symbol", "symbol_type": sym_type}

        if len(sym_text) > chunk_size * 1.8:
            sub = _recursive_chunk(
                sym_text,
                file_path,
                lang,
                chunk_size,
                overlap,
                f"{file_path}:{sym_name}:{sym['start_line']}",
                {**base_meta, "parent_symbol": sym_name},
            )
            for sc in sub:
                sc.symbol = sym_name
            chunks.extend(sub)
        else:
            chunk = CodeChunk(
                chunk_id=f"{file_path}:{sym_name}:{sym['start_line']}",
                text=sym_text,
                file_path=file_path,
                start_line=sym["start_line"],
                end_line=sym["end_line"],
                symbol=sym_name,
                language=lang,
                metadata=base_meta,
            )
            chunks.append(chunk)

        prev_end_line = sym_end_line

    if prev_end_line < len(source_lines):
        tail = "".join(source_lines[prev_end_line:]).strip()
        if tail and len(tail) > 20:
            chunks.extend(
                _recursive_chunk(
                    tail,
                    file_path,
                    lang,
                    chunk_size,
                    overlap,
                    f"{file_path}:tail",
                    {"strategy": "structural_tail"},
                )
            )

    logger.debug("tree-sitter chunked %s into %d structural chunks", file_path, len(chunks))
    return chunks


# Back-compat alias
chunk_python_file = chunk_with_tree_sitter


def chunk_text(
    text: str,
    file_path: str,
    language: str | None = None,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[CodeChunk]:
    """High quality recursive-style chunking with code-friendly separators.

    Used as fallback for non-Python files and inside oversized symbols.
    """
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap

    return _recursive_chunk(text, file_path, language, chunk_size, overlap, f"{file_path}:0")


def chunk_file(text: str, file_path: str, language: str | None = None) -> list[CodeChunk]:
    """Dispatch to the best chunker.

    For languages with tree-sitter support (python, js/ts, etc.): use structural chunking.
    Fallback: recursive text chunking.
    """
    lang = (language or "").lower()
    supported = {"python", ".py", "javascript", ".js", ".jsx", "typescript", ".ts", ".tsx"}
    if lang in supported or lang == "python":
        return chunk_with_tree_sitter(text, file_path, language=language)
    return chunk_text(text, file_path, language=language)
