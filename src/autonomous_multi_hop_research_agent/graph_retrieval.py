"""Graph-based retrieval over HotpotQA supporting-fact co-occurrence."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING

from autonomous_multi_hop_research_agent.config import HOTPOTQA_RETRIEVAL_DIR, PROCESSED_DATA_DIR
from autonomous_multi_hop_research_agent.retrieval import DenseRetriever

if TYPE_CHECKING:
    import pandas as pd


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
GENERIC_SINGLE_TOKEN_TITLES = {
    "album",
    "book",
    "film",
    "magazine",
    "movie",
    "novel",
    "series",
    "show",
    "song",
    "television",
    "women",
    "woman",
    "man",
}


def normalize_text(text: str) -> str:
    return " ".join(TOKEN_PATTERN.findall(str(text).lower()))


def normalize_title(title: str) -> str:
    return " ".join(str(title).strip().split()).lower()


@dataclass(slots=True)
class GraphRetrievalResult:
    merged_chunks: "pd.DataFrame"
    seed_titles: list[str]
    hop1_titles: list[str]
    hop2_titles: list[str]
    expanded_titles: list[str]
    fallback_reason: str = ""


@dataclass(slots=True)
class GraphArtifacts:
    nodes_path: Path
    edges_path: Path
    config_path: Path
    node_count: int
    edge_count: int
    question_count: int


def load_question_table(questions_path: Path | None = None) -> "pd.DataFrame":
    import pandas as pd

    questions_path = questions_path or (PROCESSED_DATA_DIR / "hotpotqa_distractor" / "questions.parquet")
    return pd.read_parquet(questions_path)


def load_paragraph_table(paragraphs_path: Path | None = None) -> "pd.DataFrame":
    import pandas as pd

    paragraphs_path = paragraphs_path or (PROCESSED_DATA_DIR / "hotpotqa_distractor" / "paragraphs.parquet")
    return pd.read_parquet(paragraphs_path)


def build_graph_artifacts(
    questions: "pd.DataFrame",
    paragraphs: "pd.DataFrame",
    output_dir: Path | None = None,
) -> GraphArtifacts:
    """Build a lightweight title co-occurrence graph from HotpotQA supporting facts."""
    import pandas as pd

    output_dir = output_dir or HOTPOTQA_RETRIEVAL_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    support_counts: Counter[str] = Counter()
    edge_counts: Counter[tuple[str, str]] = Counter()

    for row in questions.itertuples(index=False):
        supporting = getattr(row, "supporting_facts", [])
        unique_titles: list[str] = []
        seen_titles: set[str] = set()
        for fact in supporting:
            title = str(fact.get("title", "")).strip()
            if not title or title in seen_titles:
                continue
            seen_titles.add(title)
            unique_titles.append(title)
            support_counts[title] += 1

        for left, right in combinations(sorted(unique_titles), 2):
            edge_counts[(left, right)] += 1

    paragraph_titles = (
        paragraphs[["title"]]
        .drop_duplicates()
        .assign(normalized_title=lambda frame: frame["title"].map(normalize_title))
        .reset_index(drop=True)
    )
    paragraph_titles["support_count"] = paragraph_titles["title"].map(lambda title: int(support_counts.get(str(title), 0)))

    edge_rows: list[dict[str, object]] = []
    adjacency_degree: Counter[str] = Counter()
    for (left, right), weight in edge_counts.items():
        adjacency_degree[left] += 1
        adjacency_degree[right] += 1
        edge_rows.append(
            {
                "source_title": left,
                "target_title": right,
                "weight": int(weight),
                "question_count": int(weight),
            }
        )

    paragraph_titles["degree"] = paragraph_titles["title"].map(lambda title: int(adjacency_degree.get(str(title), 0)))
    paragraph_titles["is_graph_seed"] = paragraph_titles["degree"] > 0

    nodes_path = output_dir / "graph_nodes.parquet"
    edges_path = output_dir / "graph_edges.parquet"
    config_path = output_dir / "graph_metadata.json"

    paragraph_titles.to_parquet(nodes_path, index=False)
    pd.DataFrame(edge_rows).to_parquet(edges_path, index=False)
    config_path.write_text(
        json.dumps(
            {
                "node_count": int(len(paragraph_titles)),
                "edge_count": int(len(edge_rows)),
                "question_count": int(len(questions)),
                "source_questions_file": "questions.parquet",
                "source_paragraphs_file": "paragraphs.parquet",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return GraphArtifacts(
        nodes_path=nodes_path,
        edges_path=edges_path,
        config_path=config_path,
        node_count=int(len(paragraph_titles)),
        edge_count=int(len(edge_rows)),
        question_count=int(len(questions)),
    )


class GraphRetriever:
    """Graph-backed retriever that expands title seeds across co-supporting pages."""

    def __init__(
        self,
        base_retriever: DenseRetriever | None = None,
        graph_dir: Path | None = None,
        hop1_top_k: int = 12,
        hop2_top_k: int = 12,
        final_top_k: int = 15,
        neighbor_limit: int = 10,
        edge_decay: float = 0.72,
    ) -> None:
        import pandas as pd

        self.base_retriever = base_retriever or DenseRetriever()
        self.graph_dir = graph_dir or HOTPOTQA_RETRIEVAL_DIR
        self.hop1_top_k = hop1_top_k
        self.hop2_top_k = hop2_top_k
        self.final_top_k = final_top_k
        self.neighbor_limit = neighbor_limit
        self.edge_decay = edge_decay

        self.nodes = pd.read_parquet(self.graph_dir / "graph_nodes.parquet")
        self.edges = pd.read_parquet(self.graph_dir / "graph_edges.parquet")
        self.metadata = self.base_retriever.metadata.copy()
        self.metadata_by_title = self.metadata.drop_duplicates(subset=["title"], keep="first").set_index(
            "title", drop=False
        )

        self.normalized_title_to_title = {
            normalize_title(title): str(title)
            for title in self.nodes["title"].astype(str).tolist()
        }
        self.sorted_titles = sorted(self.normalized_title_to_title.keys(), key=len, reverse=True)
        self.adjacency: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for row in self.edges.itertuples(index=False):
            source = str(row.source_title)
            target = str(row.target_title)
            weight = float(row.weight)
            self.adjacency[source].append((target, weight))
            self.adjacency[target].append((source, weight))

        self.max_edge_weight = max(
            [weight for neighbors in self.adjacency.values() for _, weight in neighbors] or [1.0]
        )

    def _question_tokens(self, question: str) -> set[str]:
        return set(normalize_text(question).split())

    def _extract_seed_titles(self, question: str) -> list[str]:
        normalized_question = normalize_text(question)
        question_tokens = self._question_tokens(question)
        scored: list[tuple[float, str]] = []

        for normalized_title in self.sorted_titles:
            if not normalized_title or len(normalized_title) < 4:
                continue
            if normalized_title in normalized_question:
                if len(normalized_title.split()) == 1 and normalized_title in GENERIC_SINGLE_TOKEN_TITLES:
                    continue
                title = self.normalized_title_to_title[normalized_title]
                score = 1.0 + min(0.4, len(normalized_title) / 120.0)
                scored.append((score, title))
                continue

            title_tokens = set(normalized_title.split())
            if not title_tokens or len(title_tokens) > 6:
                continue
            if len(title_tokens) == 1 and next(iter(title_tokens)) in GENERIC_SINGLE_TOKEN_TITLES:
                continue
            overlap = len(title_tokens & question_tokens)
            overlap_ratio = overlap / max(1, len(title_tokens))
            if overlap >= 2 and overlap_ratio >= 0.75:
                score = 0.55 + (overlap / max(1, len(title_tokens)))
                scored.append((score, self.normalized_title_to_title[normalized_title]))

        if not scored:
            return []

        deduped: dict[str, float] = {}
        for score, title in scored:
            deduped[title] = max(deduped.get(title, 0.0), score)
        ranked_titles = [title for title, _ in sorted(deduped.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))]
        return ranked_titles[: max(8, self.hop1_top_k)]

    def _fallback_seed_titles(self, question: str, limit: int) -> list[str]:
        fallback_hits = self.base_retriever.retrieve(question, top_k=max(limit, self.final_top_k))
        if fallback_hits.empty:
            return []
        return fallback_hits["title"].astype(str).drop_duplicates().head(limit).tolist()

    def _expand_titles(self, start_titles: list[str], hop_depth: int) -> tuple[dict[str, float], list[str], list[str]]:
        ranked: dict[str, float] = {title: 1.25 for title in start_titles}
        hop1_titles: list[str] = []
        hop2_titles: list[str] = []
        visited = set(start_titles)
        frontier = list(start_titles)

        for hop in range(1, hop_depth + 1):
            next_frontier: list[str] = []
            collected: list[str] = []
            for title in frontier:
                neighbors = sorted(
                    self.adjacency.get(title, []),
                    key=lambda item: (-item[1], item[0]),
                )[: self.neighbor_limit]
                if hop == 1:
                    hop1_titles.extend([neighbor for neighbor, _ in neighbors])
                else:
                    hop2_titles.extend([neighbor for neighbor, _ in neighbors])
                for neighbor_title, weight in neighbors:
                    edge_score = (weight / self.max_edge_weight) if self.max_edge_weight else 1.0
                    candidate_score = max(ranked.get(neighbor_title, 0.0), (1.0 / hop) + (edge_score * self.edge_decay))
                    ranked[neighbor_title] = candidate_score
                    collected.append(neighbor_title)
                    if neighbor_title not in visited:
                        visited.add(neighbor_title)
                        next_frontier.append(neighbor_title)
            frontier = list(dict.fromkeys(next_frontier))
            if hop == 1 and not hop1_titles:
                hop1_titles = collected
            if hop == 2 and not hop2_titles:
                hop2_titles = collected

        return ranked, list(dict.fromkeys(hop1_titles)), list(dict.fromkeys(hop2_titles))

    def _title_rows(self, ranked_titles: dict[str, float]) -> "pd.DataFrame":
        import pandas as pd

        if not ranked_titles:
            return pd.DataFrame(columns=list(self.metadata.columns) + ["score", "graph_distance"])

        available_titles = [title for title in ranked_titles if title in self.metadata_by_title.index]
        if not available_titles:
            return pd.DataFrame(columns=list(self.metadata.columns) + ["score", "graph_distance"])

        rows = self.metadata_by_title.loc[available_titles].reset_index(drop=True).copy()
        rows["score"] = rows["title"].map(lambda title: float(ranked_titles.get(str(title), 0.0)))
        rows["graph_distance"] = rows["score"].map(lambda score: 0 if score >= 1.2 else 1 if score >= 0.9 else 2)
        rows = rows.sort_values(["score", "title"], ascending=[False, True]).drop_duplicates(subset=["title"])
        return rows[
            [
                "question_id",
                "paragraph_id",
                "title",
                "paragraph_text",
                "is_supporting_paragraph",
                "supporting_sentence_indices",
                "score",
                "graph_distance",
            ]
        ]

    def retrieve_with_debug(self, question: str, top_k: int = 15, hop_depth: int = 1) -> GraphRetrievalResult:
        import pandas as pd

        if top_k <= 0 or self.metadata.empty:
            return GraphRetrievalResult(
                merged_chunks=pd.DataFrame(
                    columns=[
                        "question_id",
                        "paragraph_id",
                        "title",
                        "paragraph_text",
                        "is_supporting_paragraph",
                        "supporting_sentence_indices",
                        "score",
                        "graph_distance",
                    ]
                ),
                seed_titles=[],
                hop1_titles=[],
                hop2_titles=[],
                expanded_titles=[],
                fallback_reason="Graph retrieval skipped because top_k was invalid or metadata was empty.",
            )

        seeds = self._extract_seed_titles(question)
        fallback_reason = ""
        if not seeds:
            seeds = self._fallback_seed_titles(question, limit=self.hop1_top_k)
            fallback_reason = "No graph titles matched the question directly; used dense fallback seeds."

        if not seeds:
            empty = pd.DataFrame(
                columns=[
                    "question_id",
                    "paragraph_id",
                    "title",
                    "paragraph_text",
                    "is_supporting_paragraph",
                    "supporting_sentence_indices",
                    "score",
                    "graph_distance",
                ]
            )
            return GraphRetrievalResult(
                merged_chunks=empty,
                seed_titles=[],
                hop1_titles=[],
                hop2_titles=[],
                expanded_titles=[],
                fallback_reason="No graph seeds were available for retrieval.",
            )

        ranked_titles, hop1_titles, hop2_titles = self._expand_titles(seeds, hop_depth=hop_depth)
        for seed in seeds:
            ranked_titles[seed] = max(ranked_titles.get(seed, 0.0), 1.25)

        ranked_rows = self._title_rows(ranked_titles)
        if ranked_rows.empty:
            return GraphRetrievalResult(
                merged_chunks=ranked_rows,
                seed_titles=seeds,
                hop1_titles=hop1_titles,
                hop2_titles=hop2_titles,
                expanded_titles=list(ranked_titles.keys()),
                fallback_reason=fallback_reason or "Graph traversal returned no matching paragraph rows.",
            )

        ranked_rows = ranked_rows.head(max(top_k, self.final_top_k)).reset_index(drop=True)
        expanded_titles = ranked_rows["title"].astype(str).tolist()
        return GraphRetrievalResult(
            merged_chunks=ranked_rows.head(top_k).reset_index(drop=True),
            seed_titles=seeds,
            hop1_titles=hop1_titles,
            hop2_titles=hop2_titles,
            expanded_titles=expanded_titles,
            fallback_reason=fallback_reason,
        )

    def retrieve(self, question: str, top_k: int = 15, hop_depth: int = 1) -> "pd.DataFrame":
        return self.retrieve_with_debug(question=question, top_k=top_k, hop_depth=hop_depth).merged_chunks
