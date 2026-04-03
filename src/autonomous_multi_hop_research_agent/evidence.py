"""Sentence-level evidence selection on top of retrieved paragraphs."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import List

from autonomous_multi_hop_research_agent.runtime import prepare_windows_torch_runtime

prepare_windows_torch_runtime()

import torch
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from autonomous_multi_hop_research_agent.config import PROCESSED_DATA_DIR
from autonomous_multi_hop_research_agent.retrieval import DEFAULT_EMBEDDING_MODEL, get_torch_device


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
logger = logging.getLogger(__name__)


def _normalize(scores: np.ndarray) -> np.ndarray:
    if len(scores) == 0:
        return scores
    min_val = scores.min()
    max_val = scores.max()
    if max_val - min_val < 1e-8:
        return np.zeros_like(scores)
    return (scores - min_val) / (max_val - min_val)


@dataclass(slots=True)
class EvidenceSelectionResult:
    selected_sentences: "pd.DataFrame"
    supporting_fact_recall: float
    matched_supporting_facts: int
    total_supporting_facts: int


def load_sentence_table(sentences_path: Path | None = None) -> "pd.DataFrame":
    """Load sentence-level rows produced in Stage 1."""
    import pandas as pd

    sentences_path = sentences_path or (PROCESSED_DATA_DIR / "hotpotqa_distractor" / "sentences.parquet")
    return pd.read_parquet(sentences_path)


class EvidenceSelector:
    """Rank sentence candidates from retrieved paragraphs against the query."""

    def __init__(
        self,
        sentences_path: Path | None = None,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        self.sentences = load_sentence_table(sentences_path=sentences_path)
        self.model_name = model_name
        device = get_torch_device()
        self.model = SentenceTransformer(model_name, device=device)
        self.reranker_model_name = RERANKER_MODEL
        self.reranker_batch_size = 32
        self.reranker: CrossEncoder | None = None
        try:
            self.reranker = CrossEncoder(self.reranker_model_name, device=device)
        except Exception as exc:  # pragma: no cover - defensive startup fallback
            logger.error("Failed to initialize cross-encoder reranker: %s", exc, exc_info=True)

    def candidate_sentences_for_retrieved(self, retrieved_chunks: "pd.DataFrame") -> "pd.DataFrame":
        """Expand retrieved paragraph hits into sentence candidates."""
        paragraph_ids = retrieved_chunks["paragraph_id"].tolist()
        candidates = self.sentences[self.sentences["paragraph_id"].isin(paragraph_ids)].copy()
        chunk_scores = retrieved_chunks[["paragraph_id", "score"]].rename(columns={"score": "chunk_score"})
        candidates = candidates.merge(chunk_scores, on="paragraph_id", how="left")
        return candidates.reset_index(drop=True)

    def score_with_reranker(self, question: str, sentences: List[str]) -> List[float]:
        """Score query-sentence pairs with a cross-encoder using batched inference."""
        if self.reranker is None:
            raise RuntimeError("Cross-encoder reranker is not available.")
        if not sentences:
            return []

        self.reranker_batch_size = min(32, max(8, len(sentences) // 4))
        pairs = [[question, sentence] for sentence in sentences]
        scores = self.reranker.predict(
            pairs,
            batch_size=self.reranker_batch_size,
            show_progress_bar=False,
        )
        return [float(score) for score in scores]

    def select_evidence(
        self,
        question: str,
        retrieved_chunks: "pd.DataFrame",
        question_id: str,
        top_k_sentences: int = 5,
    ) -> EvidenceSelectionResult:
        """Select the top-k sentence candidates and compute supporting-fact recall."""
        import pandas as pd

        candidates = self.candidate_sentences_for_retrieved(retrieved_chunks)
        if candidates.empty:
            return EvidenceSelectionResult(
                selected_sentences=pd.DataFrame(),
                supporting_fact_recall=0.0,
                matched_supporting_facts=0,
                total_supporting_facts=0,
            )

        sentence_texts = [
            f"Title: {row['title']}\nSentence: {row['sentence_text']}"
            for _, row in candidates.iterrows()
        ]
        question_embedding = self.model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        sentence_embeddings = self.model.encode(
            sentence_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        semantic_scores = (sentence_embeddings @ question_embedding[0]).astype("float32")
        candidates["semantic_score"] = semantic_scores

        semantic_norm = _normalize(candidates["semantic_score"].astype("float32").to_numpy())
        chunk_norm = _normalize(candidates["chunk_score"].fillna(0.0).astype("float32").to_numpy())
        candidates["semantic_score"] = semantic_norm
        candidates["chunk_score"] = chunk_norm
        candidates["evidence_score"] = candidates["semantic_score"] + (0.15 * candidates["chunk_score"])
        candidates["reranker_score"] = candidates["semantic_score"]
        candidates["final_score"] = (
            (0.6 * candidates["reranker_score"])
            + (0.3 * candidates["semantic_score"])
            + (0.1 * candidates["chunk_score"])
        )

        try:
            reranker_scores = self.score_with_reranker(
                question=question,
                sentences=candidates["sentence_text"].astype(str).tolist(),
            )
            reranker_norm = _normalize(np.asarray(reranker_scores, dtype="float32"))
            candidates["reranker_score"] = reranker_norm
            candidates["final_score"] = (
                (0.6 * candidates["reranker_score"])
                + (0.3 * candidates["semantic_score"])
                + (0.1 * candidates["chunk_score"])
            )
            logger.info(
                "Evidence scoring ranges after normalization | semantic=[%.3f, %.3f] chunk=[%.3f, %.3f] reranker=[%.3f, %.3f]",
                float(candidates["semantic_score"].min()),
                float(candidates["semantic_score"].max()),
                float(candidates["chunk_score"].min()),
                float(candidates["chunk_score"].max()),
                float(candidates["reranker_score"].min()),
                float(candidates["reranker_score"].max()),
            )
        except Exception as exc:  # pragma: no cover - safety fallback for runtime/model issues
            candidates["reranker_score"] = candidates["semantic_score"]
            candidates["final_score"] = (
                (0.6 * candidates["reranker_score"])
                + (0.3 * candidates["semantic_score"])
                + (0.1 * candidates["chunk_score"])
            )
            logger.error(
                "Reranker scoring failed; fallback uses normalized semantic scores as reranker_score: %s",
                exc,
                exc_info=True,
            )

        selected = (
            candidates.sort_values(["final_score", "evidence_score", "chunk_score"], ascending=False)
            .drop_duplicates(subset=["title", "sentence_text"])
            .head(top_k_sentences)
            .reset_index(drop=True)
        )

        gold = self.sentences[
            (self.sentences["question_id"] == question_id) & (self.sentences["is_supporting_fact"])
        ].copy()
        gold_ids = set(gold["sentence_id"].tolist())
        selected_ids = set(selected["sentence_id"].tolist())
        matched = len(gold_ids & selected_ids)
        total = len(gold_ids)
        recall = matched / total if total else 0.0

        return EvidenceSelectionResult(
            selected_sentences=selected[
                [
                    "question_id",
                    "paragraph_id",
                    "sentence_id",
                    "title",
                    "sentence_index",
                    "sentence_text",
                    "chunk_score",
                    "semantic_score",
                    "evidence_score",
                    "reranker_score",
                    "final_score",
                    "is_supporting_fact",
                ]
            ],
            supporting_fact_recall=recall,
            matched_supporting_facts=matched,
            total_supporting_facts=total,
        )
