"""Configuration for askgraph (local-first, privacy priority)."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """askgraph settings. All local by default."""

    model_config = SettingsConfigDict(
        env_prefix="ASKGRAPH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Storage
    index_dir: Path = Field(
        default=Path(".askgraph"),
        description="Directory (relative to target) where indexes and graphs are stored. Gitignored by default.",
    )

    # Local LLM (Ollama is the privacy-first default)
    ollama_host: str = Field(default="http://localhost:11434", description="Ollama server URL")
    default_llm_model: str = Field(
        default="llama3.2",  # or "qwen2.5:7b", "gemma2:2b", etc. Small and fast local models
        description="Default Ollama model for generation (pull it with `ollama pull <model>`)",
    )
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="Local fastembed model used for indexing and retrieval (must match between the two).",
    )

    # Retrieval
    top_k: int = Field(default=8, description="Default number of chunks to retrieve")
    lexical_alpha: float = Field(
        default=0.2,
        description=(
            "Weight of lexical (identifier/symbol-name) signal vs vector similarity in "
            "hybrid retrieval, 0..1. 0 = pure vector. Tuned to ~0.2 on the eval suite: "
            "improves ranking (MRR) without regressing recall; higher values start "
            "dropping relevant results."
        ),
    )
    chunk_size: int = Field(
        default=600, description="Target chunk size in characters for fallback chunking"
    )
    chunk_overlap: int = Field(default=80)
    max_file_bytes: int = Field(
        default=262144,  # 256 KB
        description=(
            "Skip files larger than this (bytes). Real source is rarely this big; "
            "the limit keeps machine-generated/vendored blobs (e.g. ctypes bindings) "
            "out of the index. Set to 0 to disable."
        ),
    )

    # Graph
    enable_structural_graph: bool = Field(
        default=True, description="Build tree-sitter based entity/relationship graph"
    )
    use_git_blame: bool = Field(
        default=False, description="Enrich nodes with git blame (requires GitPython extra)"
    )

    # Privacy / behavior
    local_only: bool = Field(
        default=True,
        description="When True, refuse to call any remote API unless explicitly overridden for a command.",
    )
    respect_gitignore: bool = Field(default=True)

    # Languages (expand later)
    supported_languages: list[str] = Field(
        default_factory=lambda: ["python"],  # "javascript", "typescript", etc. when parsers added
    )


settings = Settings()


def get_index_path(target_dir: Path | None = None) -> Path:
    """Return the full path to the askgraph index directory for a given target."""
    base = target_dir or Path.cwd()
    return (base / settings.index_dir).resolve()
