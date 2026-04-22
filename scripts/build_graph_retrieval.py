"""Stage 2b graph index builder for HotpotQA title co-occurrence retrieval."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from autonomous_multi_hop_research_agent.graph_retrieval import (
    build_graph_artifacts,
    load_paragraph_table,
    load_question_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a HotpotQA supporting-fact graph index.")
    parser.add_argument(
        "--questions-path",
        type=Path,
        default=None,
        help="Optional path to the questions parquet file.",
    )
    parser.add_argument(
        "--paragraphs-path",
        type=Path,
        default=None,
        help="Optional path to the paragraph parquet file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory for graph artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = load_question_table(questions_path=args.questions_path)
    paragraphs = load_paragraph_table(paragraphs_path=args.paragraphs_path)
    artifacts = build_graph_artifacts(questions=questions, paragraphs=paragraphs, output_dir=args.output_dir)

    print("Stage 2b graph build completed.")
    print(f"Node count: {artifacts.node_count}")
    print(f"Edge count: {artifacts.edge_count}")
    print(f"Question count: {artifacts.question_count}")
    print(f"Nodes path: {artifacts.nodes_path}")
    print(f"Edges path: {artifacts.edges_path}")
    print(f"Config path: {artifacts.config_path}")


if __name__ == "__main__":
    main()
