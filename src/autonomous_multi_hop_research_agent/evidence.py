"""Sentence-level evidence selection on top of retrieved paragraphs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

from autonomous_multi_hop_research_agent.config import PROCESSED_DATA_DIR
from autonomous_multi_hop_research_agent.retrieval import DEFAULT_EMBEDDING_MODEL


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
        self.model = SentenceTransformer(model_name, device="cuda")

    def candidate_sentences_for_retrieved(self, retrieved_chunks: "pd.DataFrame") -> "pd.DataFrame":
        """Expand retrieved paragraph hits into sentence candidates."""
        paragraph_ids = retrieved_chunks["paragraph_id"].tolist()
        candidates = self.sentences[self.sentences["paragraph_id"].isin(paragraph_ids)].copy()
        chunk_scores = retrieved_chunks[["paragraph_id", "score"]].rename(columns={"score": "chunk_score"})
        candidates = candidates.merge(chunk_scores, on="paragraph_id", how="left")
        return candidates.reset_index(drop=True)

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
        candidates["evidence_score"] = candidates["semantic_score"] + (0.15 * candidates["chunk_score"])

        selected = (
            candidates.sort_values(["evidence_score", "chunk_score"], ascending=False)
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
                    "is_supporting_fact",
                ]
            ],
            supporting_fact_recall=recall,
            matched_supporting_facts=matched,
            total_supporting_facts=total,
        )
