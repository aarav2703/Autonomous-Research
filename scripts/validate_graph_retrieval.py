"""Stage 2b validation script for graph-based retrieval."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from autonomous_multi_hop_research_agent.graph_retrieval import GraphRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate graph retrieval on a sample question.")
    parser.add_argument(
        "--question",
        type=str,
        default="Which magazine was started first Arthur's Magazine or First for Women?",
        help="Question to test against the graph retriever.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieved titles to print.")
    parser.add_argument(
        "--hop-depth",
        type=int,
        default=2,
        help="Graph traversal depth to use during validation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retriever = GraphRetriever()
    result = retriever.retrieve_with_debug(question=args.question, top_k=args.top_k, hop_depth=args.hop_depth)

    print("Stage 2b graph retrieval validation")
    print(f"Question: {args.question}")
    print(f"Fallback reason: {result.fallback_reason or 'none'}")
    print(f"Seed titles: {result.seed_titles}")
    print(f"Hop 1 titles: {result.hop1_titles[:10]}")
    print(f"Hop 2 titles: {result.hop2_titles[:10]}")
    print("Retrieved chunks:")
    if result.merged_chunks.empty:
        print("  <none>")
    else:
        for row in result.merged_chunks.itertuples(index=False):
            print(f"  - {row.title} | score={float(row.score):.4f} | distance={int(row.graph_distance)}")


if __name__ == "__main__":
    main()
