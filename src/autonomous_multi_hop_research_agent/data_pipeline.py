"""HotpotQA loading and preprocessing utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset

from autonomous_multi_hop_research_agent.config import PROCESSED_DATA_DIR, RAW_DATA_DIR


SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+")


@dataclass(slots=True)
class HotpotArtifacts:
    questions: pd.DataFrame
    paragraphs: pd.DataFrame
    sentences: pd.DataFrame


def simple_sentence_split(text: str) -> list[str]:
    """Split text into rough sentences without adding a heavyweight NLP dependency."""
    normalized = " ".join(text.split())
    if not normalized:
        return []
    return [part.strip() for part in SENTENCE_SPLIT_REGEX.split(normalized) if part.strip()]


def load_hotpotqa_distractor(sample_limit: int | None = None) -> Any:
    """Load the HotpotQA distractor split from Hugging Face."""
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
    if sample_limit is not None:
        dataset = dataset.select(range(min(sample_limit, len(dataset))))
    return dataset


def preprocess_hotpotqa(sample_limit: int | None = None) -> HotpotArtifacts:
    """Build traceable question, paragraph, and sentence tables from HotpotQA."""
    dataset = load_hotpotqa_distractor(sample_limit=sample_limit)

    question_rows: list[dict[str, Any]] = []
    paragraph_rows: list[dict[str, Any]] = []
    sentence_rows: list[dict[str, Any]] = []

    for row_idx, example in enumerate(dataset):
        question_id = f"hpqa_{row_idx:06d}"

        supporting_pairs = [
            {"title": title, "sentence_index": sentence_idx}
            for title, sentence_idx in zip(
                example["supporting_facts"]["title"],
                example["supporting_facts"]["sent_id"],
                strict=True,
            )
        ]

        question_rows.append(
            {
                "question_id": question_id,
                "dataset_row_id": example["id"],
                "question": example["question"],
                "answer": example["answer"],
                "type": example["type"],
                "level": example["level"],
                "supporting_facts": supporting_pairs,
            }
        )

        titles = example["context"]["title"]
        sentence_lists = example["context"]["sentences"]

        for para_idx, (title, sentences) in enumerate(zip(titles, sentence_lists, strict=True)):
            paragraph_id = f"{question_id}_p{para_idx:02d}"
            paragraph_text = " ".join(sentence.strip() for sentence in sentences if sentence.strip())
            supporting_sentence_indices = sorted(
                fact["sentence_index"] for fact in supporting_pairs if fact["title"] == title
            )

            paragraph_rows.append(
                {
                    "question_id": question_id,
                    "dataset_row_id": example["id"],
                    "paragraph_id": paragraph_id,
                    "paragraph_index": para_idx,
                    "title": title,
                    "paragraph_text": paragraph_text,
                    "sentence_count": len(sentences),
                    "supporting_sentence_indices": supporting_sentence_indices,
                    "is_supporting_paragraph": bool(supporting_sentence_indices),
                }
            )

            for sent_idx, sentence in enumerate(sentences):
                sentence_id = f"{paragraph_id}_s{sent_idx:02d}"
                normalized_sentence = " ".join(sentence.split())
                sentence_rows.append(
                    {
                        "question_id": question_id,
                        "dataset_row_id": example["id"],
                        "paragraph_id": paragraph_id,
                        "sentence_id": sentence_id,
                        "title": title,
                        "paragraph_index": para_idx,
                        "sentence_index": sent_idx,
                        "sentence_text": normalized_sentence,
                        "is_supporting_fact": sent_idx in supporting_sentence_indices,
                    }
                )

            extra_sentences = simple_sentence_split(paragraph_text)
            if len(extra_sentences) != len(sentences):
                # Keep a light signal that a regex splitter would disagree with dataset sentence boundaries.
                paragraph_rows[-1]["regex_sentence_count"] = len(extra_sentences)
            else:
                paragraph_rows[-1]["regex_sentence_count"] = len(sentences)

    return HotpotArtifacts(
        questions=pd.DataFrame(question_rows),
        paragraphs=pd.DataFrame(paragraph_rows),
        sentences=pd.DataFrame(sentence_rows),
    )


def save_artifacts(artifacts: HotpotArtifacts, output_dir: Path | None = None) -> dict[str, Path]:
    """Persist preprocessing outputs as parquet files."""
    output_dir = output_dir or PROCESSED_DATA_DIR / "hotpotqa_distractor"
    output_dir.mkdir(parents=True, exist_ok=True)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    outputs = {
        "questions": output_dir / "questions.parquet",
        "paragraphs": output_dir / "paragraphs.parquet",
        "sentences": output_dir / "sentences.parquet",
    }
    artifacts.questions.to_parquet(outputs["questions"], index=False)
    artifacts.paragraphs.to_parquet(outputs["paragraphs"], index=False)
    artifacts.sentences.to_parquet(outputs["sentences"], index=False)
    return outputs


def build_sample_report(artifacts: HotpotArtifacts, sample_index: int = 0) -> str:
    """Format one example for the validation checkpoint."""
    question_row = artifacts.questions.iloc[sample_index]
    paragraph_rows = artifacts.paragraphs[artifacts.paragraphs["question_id"] == question_row["question_id"]]
    sentence_rows = artifacts.sentences[artifacts.sentences["question_id"] == question_row["question_id"]]
    supporting_rows = sentence_rows[sentence_rows["is_supporting_fact"]]

    lines = [
        f"Question ID: {question_row['question_id']}",
        f"Question: {question_row['question']}",
        f"Answer: {question_row['answer']}",
        "Context paragraphs:",
    ]
    for _, row in paragraph_rows.iterrows():
        lines.append(
            f"  - [{row['paragraph_id']}] {row['title']}: {row['paragraph_text'][:220]}"
            + ("..." if len(row["paragraph_text"]) > 220 else "")
        )
    lines.append("Supporting facts:")
    for _, row in supporting_rows.iterrows():
        lines.append(f"  - {row['title']} | sentence {row['sentence_index']}: {row['sentence_text']}")
    return "\n".join(lines)
