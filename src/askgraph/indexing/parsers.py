"""Tree-sitter based structural parsing (extensible).

Goal (inspired by Graphify): extract real entities (Module, Function, Class, Method)
and basic relationships (imports, contains) instead of blind text chunks.
This enables better "god node" detection, neighborhood expansion, and reports.

Core language: Python (always available).
Additional languages (JS/TS, etc.) loaded if the optional tree-sitter-* packages
from the [tree-sitter-full] extra are installed.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from tree_sitter import Language, Parser, Tree

from askgraph.utils.logging import get_logger

logger = get_logger(__name__)

# Lazy-loaded languages
_python_language: Language | None = None
_js_language: Language | None = None


def get_python_language() -> Language:
    global _python_language
    if _python_language is None:
        import tree_sitter_python as tspy

        _python_language = Language(tspy.language())
    return _python_language


def get_js_language() -> Language | None:
    global _js_language
    if _js_language is None:
        try:
            import tree_sitter_javascript as tsjs

            _js_language = Language(tsjs.language())
        except ImportError:
            return None
    return _js_language


def _callee_name(fn_node: Any) -> str | None:
    """Best-effort callee name from a call's `function` node.

    `foo(...)` -> "foo"; `obj.bar(...)` / `self.bar(...)` -> "bar". Anything else
    (subscripts, calls-of-calls) returns None.
    """
    if fn_node is None:
        return None
    if fn_node.type == "identifier":
        name: str = fn_node.text.decode()
        return name
    if fn_node.type == "attribute":
        attr = fn_node.child_by_field_name("attribute")
        if attr is not None:
            attr_name: str = attr.text.decode()
            return attr_name
    return None


def _attach_calls(symbols: list[dict[str, Any]], calls_raw: list[tuple[str, int]]) -> None:
    """Attribute each call to the innermost symbol whose line range contains it."""
    for name, line in calls_raw:
        container: dict[str, Any] | None = None
        for s in symbols:
            in_range = s["start_line"] <= line <= s["end_line"]
            smaller = container is None or (s["end_line"] - s["start_line"]) < (
                container["end_line"] - container["start_line"]
            )
            if in_range and smaller:
                container = s
        if container is not None and name != container["name"]:
            container.setdefault("calls", []).append(name)
    for s in symbols:
        if "calls" in s:
            s["calls"] = sorted(set(s["calls"]))


def parse_python(source: str | bytes, path: str = "<unknown>") -> dict[str, Any]:
    """Parse Python source and return a simple structural summary + symbol list."""
    if isinstance(source, str):
        source = source.encode("utf-8")

    parser = Parser(get_python_language())
    tree: Tree = parser.parse(source)

    root = tree.root_node
    symbols: list[dict[str, Any]] = []
    imports: list[str] = []
    calls_raw: list[tuple[str, int]] = []  # (callee_name, line_no)

    def visit(node: Any, depth: int = 0) -> None:
        if node.type in ("import_statement", "import_from_statement"):
            text = node.text.decode("utf-8", errors="ignore").strip()
            imports.append(text)
        elif node.type in ("function_definition", "async_function_definition", "class_definition"):
            name_node = node.child_by_field_name("name")
            name = name_node.text.decode() if name_node else "<anon>"
            symbols.append(
                {
                    "type": "function" if "function" in node.type else "class",
                    "name": name,
                    "start_line": node.start_point[0] + 1,
                    "end_line": node.end_point[0] + 1,
                    "path": path,
                }
            )
        elif node.type == "call":
            callee = _callee_name(node.child_by_field_name("function"))
            if callee:
                calls_raw.append((callee, node.start_point[0] + 1))
        for child in node.children:
            visit(child, depth + 1)

    visit(root)
    _attach_calls(symbols, calls_raw)

    return {
        "path": path,
        "symbols": symbols,
        "imports": imports,
        "tree_sitter_ok": True,
        "language": "python",
    }


def parse_javascript(source: str | bytes, path: str = "<unknown>") -> dict[str, Any] | None:
    """Parse JS/TS (if tree-sitter-javascript is installed)."""
    lang = get_js_language()
    if lang is None:
        return None

    if isinstance(source, str):
        source = source.encode("utf-8")

    parser = Parser(lang)
    tree: Tree = parser.parse(source)

    root = tree.root_node
    symbols: list[dict[str, Any]] = []
    imports: list[str] = []

    def visit(node: Any) -> None:
        if node.type in ("import_statement", "import_from_statement"):
            text = node.text.decode("utf-8", errors="ignore").strip()
            imports.append(text)
        elif node.type in ("function_declaration", "class_declaration", "method_definition"):
            name_node = node.child_by_field_name("name")
            name = name_node.text.decode() if name_node else "<anon>"
            ntype = "class" if "class" in node.type else "function"
            symbols.append(
                {
                    "type": ntype,
                    "name": name,
                    "start_line": node.start_point[0] + 1,
                    "end_line": node.end_point[0] + 1,
                    "path": path,
                }
            )
        elif node.type == "variable_declarator":
            # Arrow-function / function-expression components: `const Foo = () => {}`,
            # `const bar = function () {}` — common in React and modern JS/TS.
            value = node.child_by_field_name("value")
            if value is not None and value.type in (
                "arrow_function",
                "function_expression",
                "function",
            ):
                name_node = node.child_by_field_name("name")
                if name_node is not None:
                    symbols.append(
                        {
                            "type": "function",
                            "name": name_node.text.decode(),
                            "start_line": node.start_point[0] + 1,
                            "end_line": node.end_point[0] + 1,
                            "path": path,
                        }
                    )
        for child in node.children:
            visit(child)

    visit(root)

    return {
        "path": path,
        "symbols": symbols,
        "imports": imports,
        "tree_sitter_ok": True,
        "language": "javascript",
    }


_LANGUAGE_PARSERS: dict[str, Callable[[str | bytes, str], dict[str, Any] | None]] = {
    "python": lambda src, p: parse_python(src, p),
    ".py": lambda src, p: parse_python(src, p),
    "javascript": lambda src, p: parse_javascript(src, p),
    ".js": lambda src, p: parse_javascript(src, p),
    ".jsx": lambda src, p: parse_javascript(src, p),
    "typescript": lambda src, p: parse_javascript(src, p),  # treat as js for now
    ".ts": lambda src, p: parse_javascript(src, p),
    ".tsx": lambda src, p: parse_javascript(src, p),
}


def _empty_parse(path: str, language: str = "unknown") -> dict[str, Any]:
    """Structural result for files we cannot parse (no symbols/imports)."""
    return {
        "path": path,
        "symbols": [],
        "imports": [],
        "tree_sitter_ok": False,
        "language": language,
    }


def parse_file(source: str | bytes, path: str = "<unknown>") -> dict[str, Any]:
    """Dispatch to the best available tree-sitter parser for the file extension.

    Returns an empty structural result for unsupported languages rather than
    mis-parsing them with the Python grammar (which would pollute the graph
    with garbage symbols for Go/Rust/Markdown/etc.).
    """
    ext = Path(path).suffix.lower()
    parser_fn = _LANGUAGE_PARSERS.get(ext) or _LANGUAGE_PARSERS.get(Path(path).name.lower())

    if parser_fn is None:
        logger.debug("No structural parser for %s — skipping symbol extraction", path)
        return _empty_parse(path, language=ext.lstrip(".") or "unknown")

    result = parser_fn(source, path)
    if result is None:
        # Parser for this language not installed (e.g. JS extra missing).
        logger.debug("Parser for %s not available — skipping symbol extraction", path)
        return _empty_parse(path, language=ext.lstrip("."))
    return result


def extract_chunks_with_symbols(source: str, path: str) -> list[dict[str, Any]]:
    """Symbol-aware chunks (used by chunking layer)."""
    parsed = parse_file(source, path)
    chunks = []
    for sym in parsed.get("symbols", []):
        chunks.append(
            {
                "symbol": sym["name"],
                "type": sym["type"],
                "start_line": sym["start_line"],
                "end_line": sym["end_line"],
                "path": path,
            }
        )
    return chunks
