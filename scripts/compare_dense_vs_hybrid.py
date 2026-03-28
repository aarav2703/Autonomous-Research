"""Compare dense-only and hybrid retrieval on the same dev sample."""

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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch

from autonomous_multi_hop_research_agent.evaluation import evaluate_dev_sample
from autonomous_multi_hop_research_agent.evidence import EvidenceSelector
from autonomous_multi_hop_research_agent.hybrid_retrieval import HybridRetriever
from autonomous_multi_hop_research_agent.retrieval import DenseRetriever
from autonomous_multi_hop_research_agent.workflow import AutonomousResearchWorkflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare dense-only and hybrid retrieval workflows.")
    parser.add_argument("--num-examples", type=int, default=100, help="Number of dev examples to evaluate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dev sampling.")
    parser.add_argument("--retrieval-top-k", type=int, default=20, help="Retrieved paragraph count.")
    parser.add_argument("--evidence-top-k", type=int, default=5, help="Selected sentence count.")
    parser.add_argument("--progress-interval", type=int, default=10, help="Progress print interval.")
    return parser.parse_args()


def failure_rank(failure_type: str) -> int:
    order = {
        "retrieval_failure": 0,
        "evidence_failure": 1,
        "generation_failure": 2,
        "success": 3,
    }
    return order.get(failure_type, -1)


def mode_score(answer_em: float, answer_f1: float, supporting_f1: float, failure_type: str) -> float:
    return 2.0 * answer_em + answer_f1 + supporting_f1 + 0.1 * failure_rank(failure_type)


def print_example(label: str, row: dict[str, object]) -> None:
    print(f"\n{label}")
    print(f"Question: {row['question']}")
    print(
        "Dense: "
        f"answer={row['predicted_answer_dense']} | "
        f"failure_type={row['failure_type_dense']} | "
        f"supporting_f1={row['supporting_f1_dense']:.2f}"
    )
    print(
        "Hybrid: "
        f"answer={row['predicted_answer_hybrid']} | "
        f"failure_type={row['failure_type_hybrid']} | "
        f"supporting_f1={row['supporting_f1_hybrid']:.2f}"
    )
    print(f"Ground truth: {row['gold_answer_dense']}")
    print("Dense cited evidence:")
    for item in row["predicted_evidence_dense"]:
        print(f"  - {item['title']} | sentence {item['sentence_index']} | {item['sentence_text']}")
    if not row["predicted_evidence_dense"]:
        print("  - None")
    print("Hybrid cited evidence:")
    for item in row["predicted_evidence_hybrid"]:
        print(f"  - {item['title']} | sentence {item['sentence_index']} | {item['sentence_text']}")
    if not row["predicted_evidence_hybrid"]:
        print("  - None")


def main() -> None:
    args = parse_args()

    dense_retriever = DenseRetriever()
    hybrid_retriever = HybridRetriever(base_retriever=dense_retriever)
    evidence_selector = EvidenceSelector()

    dense_workflow = AutonomousResearchWorkflow(
        retriever=dense_retriever,
        hybrid_retriever=hybrid_retriever,
        evidence_selector=evidence_selector,
        use_multi_hop=False,
        use_hybrid_retrieval=False,
    )
    hybrid_workflow = AutonomousResearchWorkflow(
        retriever=dense_retriever,
        hybrid_retriever=hybrid_retriever,
        evidence_selector=evidence_selector,
        use_multi_hop=False,
        use_hybrid_retrieval=True,
    )

    print("Running dense-only workflow...", flush=True)
    dense_summary, dense_results = evaluate_dev_sample(
        workflow=dense_workflow,
        num_examples=args.num_examples,
        seed=args.seed,
        retrieval_top_k=args.retrieval_top_k,
        evidence_top_k=args.evidence_top_k,
        progress_interval=args.progress_interval,
    )

    print("\nRunning hybrid workflow...", flush=True)
    hybrid_summary, hybrid_results = evaluate_dev_sample(
        workflow=hybrid_workflow,
        num_examples=args.num_examples,
        seed=args.seed,
        retrieval_top_k=args.retrieval_top_k,
        evidence_top_k=args.evidence_top_k,
        progress_interval=args.progress_interval,
    )

    comparison = dense_results.merge(
        hybrid_results,
        on=["row_index", "question_id", "question"],
        suffixes=("_dense", "_hybrid"),
    )
    comparison["dense_score"] = comparison.apply(
        lambda row: mode_score(
            answer_em=float(row["answer_em_dense"]),
            answer_f1=float(row["answer_f1_dense"]),
            supporting_f1=float(row["supporting_f1_dense"]),
            failure_type=str(row["failure_type_dense"]),
        ),
        axis=1,
    )
    comparison["hybrid_score"] = comparison.apply(
        lambda row: mode_score(
            answer_em=float(row["answer_em_hybrid"]),
            answer_f1=float(row["answer_f1_hybrid"]),
            supporting_f1=float(row["supporting_f1_hybrid"]),
            failure_type=str(row["failure_type_hybrid"]),
        ),
        axis=1,
    )
    comparison["score_delta"] = comparison["hybrid_score"] - comparison["dense_score"]

    improved_rows = comparison[comparison["score_delta"] > 0].sort_values("score_delta", ascending=False)
    unchanged_rows = comparison[
        (comparison["predicted_answer_dense"] == comparison["predicted_answer_hybrid"])
        & (comparison["failure_type_dense"] == comparison["failure_type_hybrid"])
        & (comparison["supporting_f1_dense"] == comparison["supporting_f1_hybrid"])
    ]
    worsened_rows = comparison[comparison["score_delta"] < 0].sort_values("score_delta")

    print("\nDense vs Hybrid")
    print(f"EM: {dense_summary.answer_em:.2f} -> {hybrid_summary.answer_em:.2f}")
    print(f"F1: {dense_summary.answer_f1:.2f} -> {hybrid_summary.answer_f1:.2f}")
    print(
        "Evidence Precision: "
        f"{dense_summary.supporting_precision:.2f} -> {hybrid_summary.supporting_precision:.2f}"
    )
    print(
        "Evidence Recall: "
        f"{dense_summary.supporting_recall:.2f} -> {hybrid_summary.supporting_recall:.2f}"
    )
    print(f"Evidence F1: {dense_summary.supporting_f1:.2f} -> {hybrid_summary.supporting_f1:.2f}")
    print(
        "Insufficient-context rate: "
        f"{dense_summary.insufficient_context_rate:.2f} -> {hybrid_summary.insufficient_context_rate:.2f}"
    )
    print(
        "Retrieval Failure: "
        f"{dense_summary.retrieval_failure_rate:.2%} -> {hybrid_summary.retrieval_failure_rate:.2%}"
    )
    print(
        "Evidence Failure: "
        f"{dense_summary.evidence_failure_rate:.2%} -> {hybrid_summary.evidence_failure_rate:.2%}"
    )
    print(
        "Generation Failure: "
        f"{dense_summary.generation_failure_rate:.2%} -> {hybrid_summary.generation_failure_rate:.2%}"
    )

    if not improved_rows.empty:
        print_example("Example 1. Improved by hybrid", improved_rows.iloc[0].to_dict())
    if not unchanged_rows.empty:
        print_example("Example 2. Unchanged", unchanged_rows.iloc[0].to_dict())
    if not worsened_rows.empty:
        print_example("Example 3. Worsened", worsened_rows.iloc[0].to_dict())
    else:
        print("\nExample 3. Worsened")
        print("No worsened example was found in this comparison run.")


if __name__ == "__main__":
    main()
