"""Stage 5 validation script for LangGraph orchestration."""

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

import torch

from autonomous_multi_hop_research_agent.workflow import AutonomousResearchWorkflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LangGraph autonomous research workflow.")
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Question to run through the workflow.",
    )
    parser.add_argument("--retrieval-top-k", type=int, default=5, help="Number of retrieved paragraphs.")
    parser.add_argument("--evidence-top-k", type=int, default=5, help="Number of selected evidence sentences.")
    parser.add_argument(
        "--single-hop",
        action="store_true",
        help="Disable multi-hop retrieval and use the original single-hop retriever.",
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Enable hybrid retrieval (dense + BM25 + title boosts).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workflow = AutonomousResearchWorkflow()
    result = workflow.run(
        question=args.question,
        retrieval_top_k=args.retrieval_top_k,
        evidence_top_k=args.evidence_top_k,
        use_multi_hop=not args.single_hop,
        use_hybrid_retrieval=args.hybrid,
    )

    response = result["response"]
    print("Stage 5 workflow validation")
    print(f"Question: {response['question']}")
    print(f"Normalized question: {response['normalized_question']}")
    print(f"Status: {response['status']}")
    print(f"Metadata: {response['metadata']}")
    print("\nExecution trace:")
    for step in response["execution_trace"]:
        print(f"  - {step}")
    print("\nAnswer:")
    print(response["answer"])
    print("\nReasoning:")
    for step in response["reasoning"]:
        print(f"  - {step}")
    print("\nEvidence:")
    for item in response["evidence"]:
        print(
            f"  - [{item['sentence_id']}] {item['title']} | "
            f"sentence {item['sentence_index']}: {item['sentence_text']}"
        )


if __name__ == "__main__":
    main()
