"""Hybrid dense plus BM25-style retrieval with title-aware boosts."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from autonomous_multi_hop_research_agent.config import RETRIEVAL_ARTIFACTS_DIR
from autonomous_multi_hop_research_agent.retrieval import DenseRetriever


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def normalize_text(text: str) -> str:
    return " ".join(TOKEN_PATTERN.findall(str(text).lower()))


@dataclass(slots=True)
class HybridRetrievalResult:
    merged_chunks: "pd.DataFrame"
    dense_chunks: "pd.DataFrame"
    bm25_chunks: "pd.DataFrame"
    title_boosts: list[dict[str, float | str]]


class BM25SQLiteRetriever:
    """SQLite FTS5-backed lexical retriever over titles and paragraphs."""

    def __init__(
        self,
        metadata: "pd.DataFrame",
        db_path: Path | None = None,
    ) -> None:
        self.metadata = metadata.copy()
        self.metadata_by_paragraph_id = (
            self.metadata.drop_duplicates(subset=["paragraph_id"]).set_index("paragraph_id", drop=False)
        )
        self.db_path = db_path or (RETRIEVAL_ARTIFACTS_DIR / "hotpotqa_distractor" / "paragraph_bm25.sqlite")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_index()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _ensure_index(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS paragraphs_fts
                USING fts5(
                    paragraph_id UNINDEXED,
                    title,
                    paragraph_text,
                    tokenize = 'unicode61'
                )
                """
            )
            row_count = connection.execute("SELECT count(*) FROM paragraphs_fts").fetchone()[0]
            if row_count == len(self.metadata_by_paragraph_id):
                return

            connection.execute("DELETE FROM paragraphs_fts")
            rows = (
                (
                    str(row.paragraph_id),
                    str(row.title),
                    str(row.paragraph_text),
                )
                for row in self.metadata_by_paragraph_id.itertuples()
            )
            connection.executemany(
                "INSERT INTO paragraphs_fts (paragraph_id, title, paragraph_text) VALUES (?, ?, ?)",
                rows,
            )
            connection.commit()

    def _build_match_query(self, query: str) -> str:
        tokens = [token for token in TOKEN_PATTERN.findall(query.lower()) if len(token) > 1]
        unique_tokens = list(dict.fromkeys(tokens))
        if not unique_tokens:
            return ""
        return " OR ".join(unique_tokens)

    def retrieve(self, query: str, top_k: int = 20) -> "pd.DataFrame":
        import pandas as pd

        if top_k <= 0:
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

        match_query = self._build_match_query(query)
        if not match_query:
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

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT paragraph_id, bm25(paragraphs_fts) AS rank_score
                FROM paragraphs_fts
                WHERE paragraphs_fts MATCH ?
                ORDER BY rank_score
                LIMIT ?
                """,
                (match_query, int(top_k)),
            ).fetchall()

        if not rows:
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

        paragraph_ids = [row[0] for row in rows]
        bm25_scores = {row[0]: float(-row[1]) for row in rows}
        retrieved = self.metadata_by_paragraph_id.loc[paragraph_ids].reset_index(drop=True)
        retrieved["score"] = retrieved["paragraph_id"].map(bm25_scores)
        return retrieved[
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


class HybridRetriever(DenseRetriever):
    """Hybrid retriever using dense search, lexical search, and title boosts."""

    def __init__(
        self,
        base_retriever: DenseRetriever | None = None,
        dense_top_k: int = 20,
        bm25_top_k: int = 20,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ) -> None:
        self.base_retriever = base_retriever or DenseRetriever()
        self.metadata = self.base_retriever.metadata
        self.model = self.base_retriever.model
        self.index = self.base_retriever.index
        self.model_name = self.base_retriever.model_name
        self.dense_top_k = dense_top_k
        self.bm25_top_k = bm25_top_k
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.bm25_retriever = BM25SQLiteRetriever(metadata=self.metadata)
        self.metadata_by_paragraph_id = (
            self.metadata.drop_duplicates(subset=["paragraph_id"]).set_index("paragraph_id", drop=False)
        )

    def title_boost(self, query: str, title: str) -> float:
        normalized_query = normalize_text(query)
        normalized_title = normalize_text(title)
        if not normalized_query or not normalized_title:
            return 0.0
        if normalized_query == normalized_title:
            return 0.20
        if normalized_title in normalized_query:
            return 0.12
        if normalized_query in normalized_title:
            return 0.08

        query_tokens = set(normalized_query.split())
        title_tokens = set(normalized_title.split())
        if title_tokens and len(title_tokens) <= 4:
            overlap = len(query_tokens & title_tokens) / len(title_tokens)
            if overlap >= 0.8:
                return 0.04
        return 0.0

    def retrieve_with_debug(self, query: str, top_k: int = 20) -> HybridRetrievalResult:
        import pandas as pd

        final_top_k = max(top_k, 20)
        dense_results = self.base_retriever.retrieve(query, top_k=max(final_top_k, self.dense_top_k))
        bm25_results = self.bm25_retriever.retrieve(query, top_k=max(final_top_k, self.bm25_top_k))

        if dense_results.empty and bm25_results.empty:
            empty = pd.DataFrame(
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
            return HybridRetrievalResult(
                merged_chunks=empty,
                dense_chunks=empty.copy(),
                bm25_chunks=empty.copy(),
                title_boosts=[],
            )

        merged_scores: dict[str, float] = {}
        for rank, row in enumerate(dense_results.itertuples(index=False), start=1):
            merged_scores[str(row.paragraph_id)] = merged_scores.get(str(row.paragraph_id), 0.0) + (
                self.dense_weight / (self.rrf_k + rank)
            )

        for rank, row in enumerate(bm25_results.itertuples(index=False), start=1):
            merged_scores[str(row.paragraph_id)] = merged_scores.get(str(row.paragraph_id), 0.0) + (
                self.bm25_weight / (self.rrf_k + rank)
            )

        candidate_ids = list(merged_scores.keys())
        rows = self.metadata_by_paragraph_id.loc[candidate_ids].reset_index(drop=True)
        rows["score"] = rows["paragraph_id"].map(merged_scores).astype(float)
        rows["title_boost"] = rows["title"].map(lambda title: self.title_boost(query, str(title)))
        rows["score"] = rows["score"] + rows["title_boost"]
        rows = rows.sort_values("score", ascending=False).head(final_top_k).reset_index(drop=True)
        merged = rows[
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
        title_boosts = (
            rows.loc[rows["title_boost"] > 0, ["paragraph_id", "title", "title_boost"]]
            .sort_values("title_boost", ascending=False)
            .to_dict(orient="records")
        )
        return HybridRetrievalResult(
            merged_chunks=merged,
            dense_chunks=dense_results.reset_index(drop=True),
            bm25_chunks=bm25_results.reset_index(drop=True),
            title_boosts=title_boosts,
        )

    def retrieve(self, query: str, top_k: int = 20) -> "pd.DataFrame":
        return self.retrieve_with_debug(query=query, top_k=top_k).merged_chunks
