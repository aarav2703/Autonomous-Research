"""Baseline dense retrieval utilities using sentence-transformers and FAISS."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from autonomous_multi_hop_research_agent.runtime import prepare_windows_torch_runtime

prepare_windows_torch_runtime()

import torch
from sentence_transformers import SentenceTransformer

from autonomous_multi_hop_research_agent.config import PROCESSED_DATA_DIR, RETRIEVAL_ARTIFACTS_DIR

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_torch_device() -> str:
    """Prefer CUDA when available, otherwise fall back to CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(slots=True)
class RetrievalArtifacts:
    metadata_path: Path
    index_path: Path
    config_path: Path
    model_name: str
    embedding_dim: int
    chunk_count: int


def load_paragraph_chunks(input_path: Path | None = None) -> pd.DataFrame:
    """Load paragraph-level chunks produced in Stage 1."""
    import pandas as pd

    input_path = input_path or (PROCESSED_DATA_DIR / "hotpotqa_distractor" / "paragraphs.parquet")
    paragraphs = pd.read_parquet(input_path)
    required_columns = {
        "question_id",
        "paragraph_id",
        "title",
        "paragraph_text",
        "is_supporting_paragraph",
        "supporting_sentence_indices",
    }
    missing = required_columns - set(paragraphs.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Missing required paragraph columns: {missing_str}")
    return paragraphs.copy()


def format_chunk_text(row: pd.Series) -> str:
    """Create the retrieval text representation for one paragraph chunk."""
    return f"Title: {row['title']}\nParagraph: {row['paragraph_text']}"


def embed_texts(
    texts: list[str],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 64,
) -> tuple[np.ndarray, SentenceTransformer]:
    """Embed texts into normalized float32 vectors."""
    import numpy as np

    model = SentenceTransformer(model_name, device=get_torch_device())
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype("float32"), model


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build a cosine-similarity-style FAISS index using normalized embeddings."""
    import faiss

    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings)
    return index


def build_retrieval_artifacts(
    paragraphs: pd.DataFrame,
    output_dir: Path | None = None,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 64,
) -> RetrievalArtifacts:
    """Build and persist metadata plus a FAISS index for paragraph retrieval."""
    output_dir = output_dir or (RETRIEVAL_ARTIFACTS_DIR / "hotpotqa_distractor")
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = paragraphs.copy()
    metadata["chunk_text"] = metadata.apply(format_chunk_text, axis=1)

    embeddings, model = embed_texts(
        texts=metadata["chunk_text"].tolist(),
        model_name=model_name,
        batch_size=batch_size,
    )
    index = build_faiss_index(embeddings)
    import faiss

    metadata_path = output_dir / "paragraph_metadata.parquet"
    index_path = output_dir / "paragraph_index.faiss"
    config_path = output_dir / "retrieval_metadata.json"

    metadata.to_parquet(metadata_path, index=False)
    faiss.write_index(index, str(index_path))
    config_path.write_text(
        json.dumps(
            {
                "model_name": model_name,
                "embedding_dimension": model.get_sentence_embedding_dimension(),
                "chunk_count": len(metadata),
                "index_file": index_path.name,
                "metadata_file": metadata_path.name,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return RetrievalArtifacts(
        metadata_path=metadata_path,
        index_path=index_path,
        config_path=config_path,
        model_name=model_name,
        embedding_dim=model.get_sentence_embedding_dimension(),
        chunk_count=len(metadata),
    )


class DenseRetriever:
    """Simple paragraph-level dense retriever."""

    def __init__(
        self,
        metadata_path: Path | None = None,
        index_path: Path | None = None,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        import faiss
        import pandas as pd

        metadata_path = metadata_path or (RETRIEVAL_ARTIFACTS_DIR / "hotpotqa_distractor" / "paragraph_metadata.parquet")
        index_path = index_path or (RETRIEVAL_ARTIFACTS_DIR / "hotpotqa_distractor" / "paragraph_index.faiss")

        self.metadata = pd.read_parquet(metadata_path)
        self.index = faiss.read_index(str(index_path))
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=get_torch_device())

    def retrieve(self, query: str, top_k: int = 5) -> pd.DataFrame:
        """Return the top-k paragraph chunks for a query."""
        import pandas as pd

        if top_k <= 0 or self.metadata.empty:
            return pd.DataFrame(
                columns=[
                    "question_id",
                    "paragraph_id",
                    "title",
                    "paragraph_text",
                    "is_supporting_paragraph",
                    "supporting_sentence_indices",
                    "score",
                ]
            )

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        scores, indices = self.index.search(query_embedding, top_k)

        rows = self.metadata.iloc[indices[0]].copy().reset_index(drop=True)
        rows["score"] = scores[0]
        return rows[
            [
                "question_id",
                "paragraph_id",
                "title",
                "paragraph_text",
                "is_supporting_paragraph",
                "supporting_sentence_indices",
                "score",
            ]
        ]


def build_gold_support_lookup(sentences_path: Path | None = None) -> pd.DataFrame:
    """Load only the gold supporting-fact sentences for evaluation-style inspection."""
    import pandas as pd

    sentences_path = sentences_path or (PROCESSED_DATA_DIR / "hotpotqa_distractor" / "sentences.parquet")
    sentences = pd.read_parquet(sentences_path)
    return sentences[sentences["is_supporting_fact"]].copy()
