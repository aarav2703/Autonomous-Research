"""Validation script for the iterative multi-hop retrieval layer."""

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
from datasets import load_dataset

from autonomous_multi_hop_research_agent.multi_hop_retrieval import MultiHopRetriever
from autonomous_multi_hop_research_agent.retrieval import DenseRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate multi-hop retrieval on sampled HotpotQA dev queries.")
    parser.add_argument("--num-queries", type=int, default=25, help="Number of dev queries to inspect.")
    parser.add_argument(
        "--search-limit",
        type=int,
        default=200,
        help="Number of sampled dev queries to scan when searching for a hop-2 improvement example.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling.")
    parser.add_argument("--single-hop-top-k", type=int, default=10, help="Top-k for the baseline single-hop retrieval.")
    parser.add_argument("--final-top-k", type=int, default=15, help="Final merged top-k for multi-hop retrieval.")
    return parser.parse_args()


def normalize_title(title: str) -> str:
    return " ".join(str(title).split()).strip().lower()


def main() -> None:
    args = parse_args()
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation").shuffle(seed=args.seed)
    search_count = min(max(args.num_queries, args.search_limit), len(dataset))
    sample = dataset.select(range(search_count))

    retriever = DenseRetriever()
    multi_hop_retriever = MultiHopRetriever(base_retriever=retriever, final_top_k=args.final_top_k)

    improvement_example: dict[str, object] | None = None

    for index, example in enumerate(sample, start=1):
        question = str(example["question"])
        gold_titles = {normalize_title(title) for title in example["supporting_facts"]["title"]}

        hop1 = retriever.retrieve(question, top_k=args.single_hop_top_k)
        debug = multi_hop_retriever.retrieve_with_debug(question=question, top_k=args.final_top_k)
        hop1_titles = hop1["title"].tolist()
        hop2_titles = debug.hop2_chunks["title"].tolist() if not debug.hop2_chunks.empty else []
        merged_titles = debug.merged_chunks["title"].tolist()

        merged_gold = {normalize_title(title) for title in merged_titles}
        hop1_gold = {normalize_title(title) for title in hop1_titles} & gold_titles
        added_gold_titles = sorted((gold_titles & merged_gold) - hop1_gold)

        if index <= args.num_queries:
            print(f"\nQuery {index}")
            print(f"Question: {question}")
            print(f"Gold titles: {sorted(gold_titles)}")
            print(f"Hop 1 titles: {hop1_titles}")
            print(f"Extracted entities: {debug.extracted_entities}")
            print(f"Hop 2 queries: {debug.hop2_queries}")
            print(f"Hop 2 titles: {hop2_titles}")
            print(f"Final merged titles: {merged_titles}")

        if added_gold_titles and improvement_example is None:
            improvement_example = {
                "query_index": index,
                "question": question,
                "gold_titles": sorted(gold_titles),
                "hop1_titles": hop1_titles,
                "entities": debug.extracted_entities,
                "hop2_queries": debug.hop2_queries,
                "hop2_titles": hop2_titles,
                "merged_titles": merged_titles,
                "added_gold_titles": added_gold_titles,
            }

    if improvement_example:
        print("\nExample where hop 2 added a missing document:")
        print(f"Query index in sampled dev order: {improvement_example['query_index']}")
        print(f"Question: {improvement_example['question']}")
        print(f"Gold titles: {improvement_example['gold_titles']}")
        print(f"Hop 1 titles: {improvement_example['hop1_titles']}")
        print(f"Extracted entities: {improvement_example['entities']}")
        print(f"Hop 2 queries: {improvement_example['hop2_queries']}")
        print(f"Hop 2 titles: {improvement_example['hop2_titles']}")
        print(f"Final merged titles: {improvement_example['merged_titles']}")
        print(f"Hop 2 added missing gold titles: {improvement_example['added_gold_titles']}")
    else:
        print(
            "\nNo sampled example showed hop 2 adding a missing gold title in this run, "
            f"even after scanning {search_count} dev queries."
        )


if __name__ == "__main__":
    main()
