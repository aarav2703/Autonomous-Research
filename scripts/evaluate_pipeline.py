"""Stage 6 evaluation runner."""

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
    """Help Windows resolve the local env's torch DLLs reliably."""
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


if os.name == "nt":
    prepare_windows_torch_runtime()

load_dotenv()

import torch

from autonomous_multi_hop_research_agent.evaluation import evaluate_dev_sample
from autonomous_multi_hop_research_agent.workflow import AutonomousResearchWorkflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the autonomous research workflow.")
    parser.add_argument("--num-examples", type=int, default=250, help="Number of dev examples to evaluate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dev sampling.")
    parser.add_argument("--retrieval-top-k", type=int, default=5, help="Retrieved paragraph count.")
    parser.add_argument("--evidence-top-k", type=int, default=5, help="Selected sentence count.")
    parser.add_argument("--progress-interval", type=int, default=10, help="Progress print interval.")
    return parser.parse_args()


def print_example(label: str, row: dict[str, object]) -> None:
    print(f"\n{label}")
    print(f"Question: {row['question']}")
    print(f"Predicted answer: {row['predicted_answer']}")
    print(f"Ground truth: {row['gold_answer']}")
    print(f"Failure type: {row['failure_type']}")
    print("Cited evidence:")
    for item in row["predicted_evidence"]:
        print(
            "  - "
            f"{item['title']} | sentence {item['sentence_index']} | {item['sentence_text']}"
        )
    if not row["predicted_evidence"]:
        print("  - None")
    print("Gold evidence:")
    for item in row["gold_evidence"]:
        print(
            "  - "
            f"{item['title']} | sentence {item['sentence_index']} | {item['sentence_text']}"
        )


def main() -> None:
    args = parse_args()
    workflow = AutonomousResearchWorkflow()
    summary, results = evaluate_dev_sample(
        workflow=workflow,
        num_examples=args.num_examples,
        seed=args.seed,
        retrieval_top_k=args.retrieval_top_k,
        evidence_top_k=args.evidence_top_k,
        progress_interval=args.progress_interval,
    )

    print(f"Evaluated on: {summary.num_examples} queries (HotpotQA dev)")
    print("\nAnswer:")
    print(f"- EM: {summary.answer_em:.2f}")
    print(f"- F1: {summary.answer_f1:.2f}")
    print("\nEvidence:")
    print(f"- Precision: {summary.supporting_precision:.2f}")
    print(f"- Recall: {summary.supporting_recall:.2f}")
    print(f"- F1: {summary.supporting_f1:.2f}")
    print("\nSystem:")
    print(f"- Insufficient-context rate: {summary.insufficient_context_rate:.2f}")
    print("\nFailure breakdown:")
    print(
        f"- retrieval_failure: {summary.retrieval_failure_count} "
        f"({summary.retrieval_failure_rate:.2%})"
    )
    print(
        f"- evidence_failure: {summary.evidence_failure_count} "
        f"({summary.evidence_failure_rate:.2%})"
    )
    print(
        f"- generation_failure: {summary.generation_failure_count} "
        f"({summary.generation_failure_rate:.2%})"
    )

    success_rows = results[results["failure_type"] == "success"]
    evidence_failure_rows = results[results["failure_type"] == "evidence_failure"]
    complete_failure_rows = results[results["failure_type"] == "retrieval_failure"]
    if complete_failure_rows.empty:
        complete_failure_rows = results[results["failure_type"] != "success"]

    if not success_rows.empty:
        print_example("Example 1. Full success", success_rows.iloc[0].to_dict())
    if not evidence_failure_rows.empty:
        print_example("Example 2. Partial failure", evidence_failure_rows.iloc[0].to_dict())
    if not complete_failure_rows.empty:
        print_example("Example 3. Complete failure", complete_failure_rows.iloc[0].to_dict())


if __name__ == "__main__":
    main()
