"""Stage 1 preprocessing entrypoint for HotpotQA."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from autonomous_multi_hop_research_agent.data_pipeline import (
    build_sample_report,
    preprocess_hotpotqa,
    save_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess HotpotQA distractor split.")
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Optional limit for faster local iteration.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional custom output directory for parquet files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = preprocess_hotpotqa(sample_limit=args.sample_limit)
    output_paths = save_artifacts(artifacts, output_dir=args.output_dir)

    print("Stage 1 preprocessing completed.")
    print(f"Questions: {len(artifacts.questions)}")
    print(f"Paragraphs: {len(artifacts.paragraphs)}")
    print(f"Sentences: {len(artifacts.sentences)}")
    print("Saved parquet files:")
    for name, path in output_paths.items():
        print(f"  - {name}: {path}")

    print("\nSample question entries:")
    sample_questions = artifacts.questions[["question_id", "question", "answer"]].head(3)
    print(sample_questions.to_string(index=False))

    print("\nValidation example:")
    print(build_sample_report(artifacts, sample_index=0))


if __name__ == "__main__":
    main()
