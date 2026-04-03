"""Stage 6 evaluation utilities for answer and supporting-fact metrics."""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from autonomous_multi_hop_research_agent.workflow import AutonomousResearchWorkflow


def normalize_answer(text: str) -> str:
    """Normalize text in a Hotpot/SQuAD-style way for EM/F1."""

    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    def lower(value: str) -> str:
        return value.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def token_f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def prf1(predicted_items: set[tuple[str, int]], gold_items: set[tuple[str, int]]) -> tuple[float, float, float]:
    if not predicted_items and not gold_items:
        return 1.0, 1.0, 1.0
    if not predicted_items:
        return 0.0, 0.0, 0.0
    if not gold_items:
        return 0.0, 0.0, 0.0

    true_positive = len(predicted_items & gold_items)
    precision = true_positive / len(predicted_items)
    recall = true_positive / len(gold_items)
    if precision + recall == 0:
        return precision, recall, 0.0
    return precision, recall, 2 * precision * recall / (precision + recall)


def normalize_title(title: str) -> str:
    return " ".join(str(title).strip().split()).lower()


def supporting_fact_keys_from_example(example: dict[str, Any]) -> set[tuple[str, int]]:
    titles = example["supporting_facts"]["title"]
    sentence_indices = example["supporting_facts"]["sent_id"]
    return {
        (normalize_title(title), int(sentence_index))
        for title, sentence_index in zip(titles, sentence_indices)
    }


def supporting_titles_from_example(example: dict[str, Any]) -> set[str]:
    return {normalize_title(title) for title in example["supporting_facts"]["title"]}


def predicted_fact_keys_from_response(response: dict[str, Any]) -> set[tuple[str, int]]:
    return {
        (normalize_title(item["title"]), int(item["sentence_index"]))
        for item in response["evidence"]
    }


def retrieved_titles_from_state(result: dict[str, Any]) -> set[str]:
    return {
        normalize_title(chunk["title"])
        for chunk in result.get("retrieved_chunks", [])
    }


def classify_failure(
    answer_em: float,
    predicted_fact_keys: set[tuple[str, int]],
    gold_fact_keys: set[tuple[str, int]],
    retrieved_titles: set[str],
    gold_titles: set[str],
) -> str:
    gold_titles_retrieved = gold_titles.issubset(retrieved_titles)
    full_evidence = predicted_fact_keys == gold_fact_keys and len(gold_fact_keys) > 0

    if answer_em == 1.0 and full_evidence:
        return "success"
    if not gold_titles_retrieved:
        return "retrieval_failure"
    if not gold_fact_keys.issubset(predicted_fact_keys):
        return "evidence_failure"
    return "generation_failure"


def format_gold_evidence(example: dict[str, Any]) -> list[dict[str, Any]]:
    context_title_to_sentences = {
        title: sentences
        for title, sentences in zip(example["context"]["title"], example["context"]["sentences"])
    }
    gold_rows: list[dict[str, Any]] = []
    for title, sentence_index in zip(
        example["supporting_facts"]["title"],
        example["supporting_facts"]["sent_id"],
    ):
        sentences = context_title_to_sentences.get(title, [])
        sentence_text = ""
        if 0 <= int(sentence_index) < len(sentences):
            sentence_text = sentences[int(sentence_index)]
        gold_rows.append(
            {
                "title": title,
                "sentence_index": int(sentence_index),
                "sentence_text": sentence_text,
            }
        )
    return gold_rows


@dataclass(slots=True)
class EvaluationSummary:
    num_examples: int
    answer_em: float
    answer_f1: float
    supporting_precision: float
    supporting_recall: float
    supporting_f1: float
    insufficient_context_rate: float
    retrieval_failure_rate: float
    evidence_failure_rate: float
    generation_failure_rate: float
    retrieval_failure_count: int
    evidence_failure_count: int
    generation_failure_count: int
    retrieval_recall_estimate: float
    evidence_recall_estimate: float
    used_multi_hop_rate: float
    used_subqueries_rate: float


def load_dev_sample(num_examples: int = 250, seed: int = 42) -> list[dict[str, Any]]:
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    shuffled = dataset.shuffle(seed=seed)
    return list(shuffled.select(range(min(num_examples, len(shuffled)))))


def evaluate_dev_sample(
    workflow: AutonomousResearchWorkflow,
    num_examples: int = 250,
    seed: int = 42,
    retrieval_top_k: int = 5,
    evidence_top_k: int = 5,
    progress_interval: int = 10,
) -> tuple[EvaluationSummary, "pd.DataFrame"]:
    """Run full-workflow evaluation on a sampled HotpotQA dev subset."""
    import pandas as pd

    examples = load_dev_sample(num_examples=num_examples, seed=seed)
    rows: list[dict[str, Any]] = []

    for idx, example in enumerate(examples, start=1):
        example_id = str(example.get("_id", example.get("id", "")))
        result = workflow.run(
            question=str(example["question"]),
            question_id=example_id,
            retrieval_top_k=retrieval_top_k,
            evidence_top_k=evidence_top_k,
        )
        response = result["response"]
        predicted_answer = str(response["answer"])
        gold_answer = str(example["answer"])

        gold_fact_keys = supporting_fact_keys_from_example(example)
        gold_titles = supporting_titles_from_example(example)
        predicted_fact_keys = predicted_fact_keys_from_response(response)
        retrieved_titles = retrieved_titles_from_state(result)
        support_precision, support_recall, support_f1 = prf1(predicted_fact_keys, gold_fact_keys)
        answer_em = exact_match_score(predicted_answer, gold_answer)
        answer_f1 = token_f1_score(predicted_answer, gold_answer)
        failure_type = classify_failure(
            answer_em=answer_em,
            predicted_fact_keys=predicted_fact_keys,
            gold_fact_keys=gold_fact_keys,
            retrieved_titles=retrieved_titles,
            gold_titles=gold_titles,
        )

        rows.append(
            {
                "row_index": idx,
                "question_id": example_id,
                "question": str(example["question"]),
                "gold_answer": gold_answer,
                "predicted_answer": predicted_answer,
                "answer_em": answer_em,
                "answer_f1": answer_f1,
                "supporting_precision": support_precision,
                "supporting_recall": support_recall,
                "supporting_f1": support_f1,
                "status": response["status"],
                "failure_reason": response["metadata"]["failure_reason"],
                "failure_type": failure_type,
                "retrieved_titles": sorted(retrieved_titles),
                "gold_titles": sorted(gold_titles),
                "predicted_evidence": response["evidence"],
                "gold_evidence": format_gold_evidence(example),
                "num_citations": len(response["evidence"]),
                "execution_trace": response["execution_trace"],
                "retrieval_recall_estimate": float(response["metadata"].get("retrieval_recall_estimate", 0.0)),
                "evidence_recall_estimate": float(response["metadata"].get("evidence_recall_estimate", 0.0)),
                "used_multi_hop": str(response["metadata"].get("used_multi_hop", "False")).lower() == "true",
                "used_subqueries": str(response["metadata"].get("used_subqueries", "False")).lower() == "true",
            }
        )
        if progress_interval > 0 and (idx % progress_interval == 0 or idx == len(examples)):
            print(
                f"[progress] evaluated {idx}/{len(examples)} dev queries "
                f"({(idx / len(examples)) * 100:.0f}%)",
                flush=True,
            )

    results = pd.DataFrame(rows)
    retrieval_failure_count = int((results["failure_type"] == "retrieval_failure").sum())
    evidence_failure_count = int((results["failure_type"] == "evidence_failure").sum())
    generation_failure_count = int((results["failure_type"] == "generation_failure").sum())

    summary = EvaluationSummary(
        num_examples=len(results),
        answer_em=float(results["answer_em"].mean()),
        answer_f1=float(results["answer_f1"].mean()),
        supporting_precision=float(results["supporting_precision"].mean()),
        supporting_recall=float(results["supporting_recall"].mean()),
        supporting_f1=float(results["supporting_f1"].mean()),
        insufficient_context_rate=float((results["status"] == "insufficient_context").mean()),
        retrieval_failure_rate=retrieval_failure_count / len(results),
        evidence_failure_rate=evidence_failure_count / len(results),
        generation_failure_rate=generation_failure_count / len(results),
        retrieval_failure_count=retrieval_failure_count,
        evidence_failure_count=evidence_failure_count,
        generation_failure_count=generation_failure_count,
        retrieval_recall_estimate=float(results["retrieval_recall_estimate"].mean()),
        evidence_recall_estimate=float(results["evidence_recall_estimate"].mean()),
        used_multi_hop_rate=float(results["used_multi_hop"].mean()),
        used_subqueries_rate=float(results["used_subqueries"].mean()),
    )
    return summary, results
