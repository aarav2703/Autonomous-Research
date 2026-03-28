"""Stage 3 validation script for evidence selection."""

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

from autonomous_multi_hop_research_agent.evidence import EvidenceSelector
from autonomous_multi_hop_research_agent.retrieval import DenseRetriever, build_gold_support_lookup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate sentence-level evidence selection.")
    parser.add_argument("--retrieval-top-k", type=int, default=5, help="Number of retrieved paragraphs per question.")
    parser.add_argument("--evidence-top-k", type=int, default=5, help="Number of selected sentences per question.")
    parser.add_argument("--num-questions", type=int, default=3, help="Number of questions to validate.")
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
    selector = EvidenceSelector()

    print("Stage 3 evidence selection validation")
    print(f"Retrieval top-k: {args.retrieval_top_k}")
    print(f"Evidence top-k: {args.evidence_top_k}")

    recall_values: list[float] = []

    for _, question_row in questions.head(args.num_questions).iterrows():
        question_id = question_row["question_id"]
        retrieved = retriever.retrieve(question_row["question"], top_k=args.retrieval_top_k)
        evidence = selector.select_evidence(
            question=question_row["question"],
            retrieved_chunks=retrieved,
            question_id=question_id,
            top_k_sentences=args.evidence_top_k,
        )
        gold = gold_sentences[gold_sentences["question_id"] == question_id]
        recall_values.append(evidence.supporting_fact_recall)

        print("\n" + "=" * 100)
        print(f"Question ID: {question_id}")
        print(f"Question: {question_row['question']}")
        print("Retrieved chunks:")
        for rank, (_, row) in enumerate(retrieved.iterrows(), start=1):
            preview = row["paragraph_text"][:180] + ("..." if len(row["paragraph_text"]) > 180 else "")
            print(
                f"  {rank}. [{row['paragraph_id']}] score={row['score']:.4f} | "
                f"title={row['title']} | supporting={row['is_supporting_paragraph']}"
            )
            print(f"     {preview}")

        print("Selected evidence sentences:")
        for rank, (_, row) in enumerate(evidence.selected_sentences.iterrows(), start=1):
            print(
                f"  {rank}. [{row['sentence_id']}] evidence={row['evidence_score']:.4f} | "
                f"chunk={row['chunk_score']:.4f} | semantic={row['semantic_score']:.4f} | "
                f"supporting={row['is_supporting_fact']}"
            )
            print(f"     {row['title']} | sentence {row['sentence_index']}: {row['sentence_text']}")

        print("Ground truth supporting facts:")
        for _, row in gold.iterrows():
            print(f"  - [{row['sentence_id']}] {row['title']} | sentence {row['sentence_index']}: {row['sentence_text']}")

        print(
            "Supporting fact recall: "
            f"{evidence.matched_supporting_facts}/{evidence.total_supporting_facts} "
            f"({evidence.supporting_fact_recall:.2f})"
        )

    if recall_values:
        avg_recall = sum(recall_values) / len(recall_values)
        print("\nAverage supporting fact recall:")
        print(f"  {avg_recall:.2f}")


if __name__ == "__main__":
    main()
