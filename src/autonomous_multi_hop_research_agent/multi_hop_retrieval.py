"""Iterative multi-hop retrieval built on top of the existing dense retriever."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from autonomous_multi_hop_research_agent.retrieval import DenseRetriever


ENTITY_PATTERN = re.compile(r"\b[A-Z][A-Za-z0-9'’.-]*(?:\s+[A-Z][A-Za-z0-9'’.-]*){0,3}\b")
GENERIC_ENTITY_BLACKLIST = {
    "The",
    "A",
    "An",
    "He",
    "She",
    "It",
    "They",
    "This",
    "That",
    "These",
    "Those",
    "One",
    "Two",
    "Three",
}


@dataclass(slots=True)
class MultiHopRetrievalResult:
    merged_chunks: "pd.DataFrame"
    hop1_chunks: "pd.DataFrame"
    hop2_chunks: "pd.DataFrame"
    extracted_entities: list[str]
    hop2_queries: list[str]
    fallback_reason: str = ""


class MultiHopRetriever:
    """Two-hop dense retrieval that expands the query with retrieved entities."""

    def __init__(
        self,
        base_retriever: DenseRetriever | None = None,
        hop1_top_k: int = 12,
        hop2_top_k: int = 4,
        final_top_k: int = 15,
        max_entities: int = 4,
    ) -> None:
        self.base_retriever = base_retriever or DenseRetriever()
        self.hop1_top_k = hop1_top_k
        self.hop2_top_k = hop2_top_k
        self.final_top_k = final_top_k
        self.max_entities = max_entities

    def _normalize_entity_key(self, entity: str) -> str:
        return " ".join(entity.split()).strip().lower()

    def _extract_text_entities(self, text: str) -> list[str]:
        matches = [match.group(0).strip(".,;:()[]{}\"'") for match in ENTITY_PATTERN.finditer(text)]
        cleaned: list[str] = []
        for match in matches:
            if len(match) < 3 or match in GENERIC_ENTITY_BLACKLIST:
                continue
            cleaned.append(match)
        return cleaned

    def extract_entities(
        self,
        retrieved_chunks: "pd.DataFrame",
        question: str = "",
        max_entities: int | None = None,
    ) -> list[str]:
        """Use retrieved titles and simple capitalized spans for query expansion."""
        limit = max_entities or self.max_entities
        if retrieved_chunks.empty or limit <= 0:
            return []

        question_key = self._normalize_entity_key(question)
        title_candidates: list[str] = []
        seen_keys: set[str] = set()

        for row in retrieved_chunks.itertuples(index=False):
            title = str(row.title).strip()
            title_key = self._normalize_entity_key(title)
            if title and title_key not in seen_keys:
                seen_keys.add(title_key)
                title_candidates.append(title)

        text_counts: dict[str, int] = {}
        for row in retrieved_chunks.itertuples(index=False):
            for entity in self._extract_text_entities(str(row.paragraph_text)):
                entity_key = self._normalize_entity_key(entity)
                if entity_key in seen_keys or (question_key and entity_key in question_key):
                    continue
                text_counts[entity] = text_counts.get(entity, 0) + 1

        ranked_text_entities = sorted(
            text_counts.items(),
            key=lambda item: (-item[1], -len(item[0]), item[0]),
        )
        selected: list[str] = []
        max_title_entities = min(2, limit)

        for title in title_candidates:
            if len(selected) >= max_title_entities:
                break
            selected.append(title)

        for entity, _ in ranked_text_entities:
            entity_key = self._normalize_entity_key(entity)
            if entity_key in {self._normalize_entity_key(item) for item in selected}:
                continue
            selected.append(entity)
            if len(selected) >= limit:
                return selected

        for title in title_candidates:
            title_key = self._normalize_entity_key(title)
            if title_key in {self._normalize_entity_key(item) for item in selected}:
                continue
            selected.append(title)
            if len(selected) >= limit:
                break

        return selected[:limit]

    def build_expanded_queries(self, question: str, entities: list[str]) -> list[str]:
        return [f"{question} {entity}".strip() for entity in entities]

    def merge_chunks(
        self,
        hop1_chunks: "pd.DataFrame",
        hop2_chunks: "pd.DataFrame",
        final_top_k: int,
    ) -> "pd.DataFrame":
        import pandas as pd

        all_chunks = pd.concat([hop1_chunks, hop2_chunks], ignore_index=True)
        if all_chunks.empty:
            return all_chunks

        merged = (
            all_chunks.sort_values("score", ascending=False)
            .drop_duplicates(subset=["paragraph_id"], keep="first")
            .head(final_top_k)
            .reset_index(drop=True)
        )
        return merged[
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

    def retrieve_with_debug(self, question: str, top_k: int = 15) -> MultiHopRetrievalResult:
        import pandas as pd

        final_top_k = max(top_k, self.final_top_k)
        hop1_chunks = self.base_retriever.retrieve(question, top_k=max(self.hop1_top_k, min(final_top_k, 15)))
        entities = self.extract_entities(hop1_chunks, question=question)
        if not entities:
            return MultiHopRetrievalResult(
                merged_chunks=hop1_chunks.head(final_top_k).reset_index(drop=True),
                hop1_chunks=hop1_chunks.reset_index(drop=True),
                hop2_chunks=pd.DataFrame(columns=hop1_chunks.columns),
                extracted_entities=[],
                hop2_queries=[],
                fallback_reason="No expansion entities extracted from hop 1; fell back to single-hop retrieval.",
            )

        hop2_queries = self.build_expanded_queries(question, entities)
        hop2_frames: list[pd.DataFrame] = []
        for expanded_query in hop2_queries:
            hop2_frame = self.base_retriever.retrieve(expanded_query, top_k=self.hop2_top_k).copy()
            if not hop2_frame.empty:
                hop2_frame["score"] = hop2_frame["score"] * 0.98
            hop2_frames.append(hop2_frame)

        hop2_chunks = (
            pd.concat(hop2_frames, ignore_index=True)
            if hop2_frames
            else pd.DataFrame(columns=hop1_chunks.columns)
        )
        merged_chunks = self.merge_chunks(
            hop1_chunks=hop1_chunks,
            hop2_chunks=hop2_chunks,
            final_top_k=final_top_k,
        )
        return MultiHopRetrievalResult(
            merged_chunks=merged_chunks,
            hop1_chunks=hop1_chunks.reset_index(drop=True),
            hop2_chunks=hop2_chunks.reset_index(drop=True),
            extracted_entities=entities,
            hop2_queries=hop2_queries,
        )

    def retrieve(self, question: str, top_k: int = 15) -> "pd.DataFrame":
        """Return merged multi-hop chunks using the same schema as DenseRetriever."""
        return self.retrieve_with_debug(question=question, top_k=top_k).merged_chunks
