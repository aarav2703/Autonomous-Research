"""Stage 4 validation script for grounded answer generation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

def prepare_windows_torch_runtime() -> None:
    if os.name != "nt":
        return

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env_root = PROJECT_ROOT / ".conda" / "envs" / "autonomous-multi-hop-research-agent"
    dll_paths = [
        env_root / "Lib" / "site-packages" / "torch" / "lib",
        env_root / "Library" / "bin",
    ]
    existing = os.environ.get("PATH", "")
    prefix_parts = [str(path) for path in dll_paths if path.exists()]
    if prefix_parts:
        os.environ["PATH"] = ";".join(prefix_parts + [existing])
    for path in dll_paths:
        if path.exists() and hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(path))


prepare_windows_torch_runtime()

load_dotenv()

from autonomous_multi_hop_research_agent.evidence import EvidenceSelector
from autonomous_multi_hop_research_agent.rag import (
    LLMConfigurationError,
    LLMRequestError,
    generate_grounded_answer,
)
from autonomous_multi_hop_research_agent.retrieval import DenseRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate grounded RAG answer generation.")
    parser.add_argument("--question-index", type=int, default=0, help="Question row index to validate.")
    parser.add_argument("--retrieval-top-k", type=int, default=5, help="Number of retrieved paragraphs.")
    parser.add_argument("--evidence-top-k", type=int, default=5, help="Number of evidence sentences.")
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
    question_row = questions.iloc[args.question_index]

    retriever = DenseRetriever()
    selector = EvidenceSelector()

    retrieved = retriever.retrieve(question_row["question"], top_k=args.retrieval_top_k)
    evidence = selector.select_evidence(
        question=question_row["question"],
        retrieved_chunks=retrieved,
        question_id=question_row["question_id"],
        top_k_sentences=args.evidence_top_k,
    )

    try:
        grounded = generate_grounded_answer(
            question=question_row["question"],
            retrieved_chunks=retrieved,
            selected_sentences=evidence.selected_sentences,
        )
    except (LLMConfigurationError, LLMRequestError) as exc:
        print("Stage 4 validation blocked.")
        print(str(exc))
        return

    print("Stage 4 grounded generation validation")
    print(f"Question ID: {question_row['question_id']}")
    print(f"Question: {question_row['question']}")
    print(f"Ground truth answer: {question_row['answer']}")
    print("\nInput prompt:")
    print(grounded.prompt)
    print("\nOutput answer:")
    print(grounded.answer)
    print("\nReasoning trace:")
    for step in grounded.reasoning_trace:
        print(f"  - {step}")
    print("\nCited evidence:")
    for item in grounded.cited_evidence:
        print(
            f"  - [{item['sentence_id']}] {item['title']} | "
            f"sentence {item['sentence_index']}: {item['sentence_text']}"
        )
    print("\nMatches ground truth:")
    print(str(grounded.answer.strip().lower() == str(question_row["answer"]).strip().lower()))


if __name__ == "__main__":
    main()
