"""Stage 2 validation script for dense retrieval."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if os.name == "nt":
    # Avoid Windows OpenMP runtime conflicts between torch and numerical deps.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from autonomous_multi_hop_research_agent.retrieval import DenseRetriever, build_gold_support_lookup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate dense retrieval against HotpotQA supporting facts.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieved chunks per question.")
    parser.add_argument(
        "--num-questions",
        type=int,
        default=3,
        help="Number of questions to print for qualitative validation.",
    )
    parser.add_argument(
        "--questions-path",
        type=Path,
        default=Path("data/processed/hotpotqa_distractor/questions.parquet"),
        help="Path to the questions parquet file.",
    )
    return parser.parse_args()


def main() -> None:
    import pandas as pd

    args = parse_args()
    questions = pd.read_parquet(args.questions_path)
    gold_sentences = build_gold_support_lookup()
    retriever = DenseRetriever()

    print("Stage 2 retrieval validation")
    print(f"Top-k: {args.top_k}")

    for _, question_row in questions.head(args.num_questions).iterrows():
        question_id = question_row["question_id"]
        retrieved = retriever.retrieve(question_row["question"], top_k=args.top_k)
        gold = gold_sentences[gold_sentences["question_id"] == question_id]

        print("\n" + "=" * 100)
        print(f"Question ID: {question_id}")
        print(f"Question: {question_row['question']}")
        print(f"Answer: {question_row['answer']}")
        print("Retrieved chunks:")
        for rank, (_, row) in enumerate(retrieved.iterrows(), start=1):
            preview = row["paragraph_text"][:220] + ("..." if len(row["paragraph_text"]) > 220 else "")
            print(
                f"  {rank}. [{row['paragraph_id']}] score={row['score']:.4f} | "
                f"title={row['title']} | supporting={row['is_supporting_paragraph']}"
            )
            print(f"     {preview}")

        print("Gold supporting facts:")
        for _, row in gold.iterrows():
            print(f"  - {row['title']} | sentence {row['sentence_index']}: {row['sentence_text']}")


if __name__ == "__main__":
    main()
