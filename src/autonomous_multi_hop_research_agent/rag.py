"""Grounded RAG generation using retrieved chunks and selected evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, List
from urllib import error, request

from autonomous_multi_hop_research_agent.config import get_llm_settings


@dataclass(slots=True)
class GroundedAnswer:
    answer: str
    reasoning_trace: list[str]
    cited_evidence: list[dict[str, Any]]
    raw_response_text: str
    prompt: str
    status: str = "ok"
    failure_reason: str = ""


class LLMConfigurationError(RuntimeError):
    """Raised when no usable LLM configuration is available."""


class LLMRequestError(RuntimeError):
    """Raised when the provider request fails or returns malformed output."""


class CitationValidationError(LLMRequestError):
    """Raised when the model cites sentence IDs outside the selected evidence set."""


def build_safe_failure_answer(
    reason: str,
    prompt: str = "",
    raw_response_text: str = "",
) -> GroundedAnswer:
    """Create a safe closed response for empty context or validation failures."""
    return GroundedAnswer(
        answer="INSUFFICIENT_CONTEXT",
        reasoning_trace=[reason],
        cited_evidence=[],
        raw_response_text=raw_response_text,
        prompt=prompt,
        status="insufficient_context",
        failure_reason=reason,
    )


def build_grounding_context(
    retrieved_chunks: "pd.DataFrame",
    selected_sentences: "pd.DataFrame",
) -> str:
    """Format retrieval and evidence state into a prompt-ready context block."""
    if retrieved_chunks.empty or selected_sentences.empty:
        return ""

    chunk_lines = ["Retrieved paragraphs:"]
    for _, row in retrieved_chunks.iterrows():
        chunk_lines.append(
            f"- paragraph_id={row['paragraph_id']} | title={row['title']} | score={row['score']:.4f}"
        )
        chunk_lines.append(f"  text: {row['paragraph_text']}")

    evidence_lines = ["Selected evidence sentences:"]
    for _, row in selected_sentences.iterrows():
        evidence_lines.append(
            f"- sentence_id={row['sentence_id']} | title={row['title']} | sentence_index={row['sentence_index']} "
            f"| evidence_score={row['evidence_score']:.4f}"
        )
        evidence_lines.append(f"  text: {row['sentence_text']}")

    return "\n".join(chunk_lines + [""] + evidence_lines)


def build_grounded_prompt(question: str, context_block: str) -> str:
    """Construct the strict grounded-answer prompt."""
    return f"""You are a grounded multi-hop QA assistant.

Rules:
1. Use only the provided context.
2. If the context is insufficient, answer exactly: INSUFFICIENT_CONTEXT
3. Do not use outside knowledge.
4. Keep the reasoning trace short and factual.
5. Cite only sentence_id values that appear in the provided evidence list.

Return valid JSON with this exact schema:
{{
  "answer": "string",
  "reasoning_trace": ["step 1", "step 2"],
  "cited_sentence_ids": ["sentence_id_1", "sentence_id_2"]
}}

Question:
{question}

Context:
{context_block}
"""


def build_subquery_prompt(question: str, context_titles: List[str]) -> str:
        """Construct a strict JSON prompt for follow-up retrieval query generation."""
        known_titles = "\n".join(f"- {title}" for title in context_titles[:20]) if context_titles else "- None"
        return f"""Question: {question}
Known context titles: {known_titles}

Task:

* Identify what information is missing to answer the question
* Generate 1–2 focused search queries that target:

    * a bridge entity OR
    * a missing comparison attribute

Rules:

* Avoid restating the original question
* Keep queries short and specific
* No explanations

Return JSON:
{{
    "subqueries": ["...", "..."]
}}
"""


class OpenAICompatibleChatClient:
    """Minimal client for OpenAI-compatible chat completion APIs."""

    def __init__(self) -> None:
        settings = get_llm_settings()
        if not settings["api_key"]:
            raise LLMConfigurationError(
                "Missing API key. Set DEEPSEEK_API_KEY or OPENAI_API_KEY before Stage 4 validation."
            )
        self.api_key = settings["api_key"]
        self.base_url = settings["base_url"].rstrip("/")
        self.model = settings["model"]

    def create_chat_completion(self, prompt: str) -> str:
        """Call the configured provider and return the assistant text."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You answer only from supplied evidence."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "stream": False,
        }
        body = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            url=f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with request.urlopen(http_request, timeout=90) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LLMRequestError(f"LLM request failed with HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise LLMRequestError(f"LLM request failed: {exc.reason}") from exc

        try:
            return response_payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMRequestError(f"Malformed provider response: {response_payload}") from exc

    def create_structured_chat_completion(self, system_prompt: str, user_prompt: str) -> str:
        """Call the provider with explicit system/user messages for structured tasks."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "stream": False,
        }
        body = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            url=f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with request.urlopen(http_request, timeout=90) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LLMRequestError(f"LLM request failed with HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise LLMRequestError(f"LLM request failed: {exc.reason}") from exc

        try:
            return response_payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMRequestError(f"Malformed provider response: {response_payload}") from exc


def parse_grounded_response(
    response_text: str,
    selected_sentences: "pd.DataFrame",
    prompt: str,
) -> GroundedAnswer:
    """Parse and validate the model's structured response."""
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise LLMRequestError(f"Model response was not valid JSON: {response_text}") from exc

    answer = payload.get("answer", "")
    reasoning_trace = payload.get("reasoning_trace", [])
    cited_sentence_ids = payload.get("cited_sentence_ids", [])

    if not isinstance(answer, str) or not isinstance(reasoning_trace, list) or not isinstance(cited_sentence_ids, list):
        raise LLMRequestError(f"Model response had the wrong schema: {payload}")

    evidence_lookup = selected_sentences.set_index("sentence_id").to_dict("index")
    invalid_ids = [sentence_id for sentence_id in cited_sentence_ids if sentence_id not in evidence_lookup]
    if invalid_ids:
        raise CitationValidationError(
            "Model cited sentence IDs not present in selected evidence: " + ", ".join(invalid_ids)
        )

    cited_evidence = []
    for sentence_id in cited_sentence_ids:
        row = evidence_lookup[sentence_id]
        cited_evidence.append(
            {
                "sentence_id": sentence_id,
                "title": row["title"],
                "sentence_index": int(row["sentence_index"]),
                "sentence_text": row["sentence_text"],
            }
        )

    return GroundedAnswer(
        answer=answer,
        reasoning_trace=[str(step) for step in reasoning_trace],
        cited_evidence=cited_evidence,
        raw_response_text=response_text,
        prompt=prompt,
    )


def generate_grounded_answer(
    question: str,
    retrieved_chunks: "pd.DataFrame",
    selected_sentences: "pd.DataFrame",
    client: OpenAICompatibleChatClient | None = None,
) -> GroundedAnswer:
    """Generate a grounded answer, refusing to answer without context."""
    context_block = build_grounding_context(retrieved_chunks, selected_sentences)
    if not context_block.strip():
        return build_safe_failure_answer(
            reason="No retrieved context or evidence was supplied, so the system refused to answer.",
        )

    prompt = build_grounded_prompt(question=question, context_block=context_block)
    client = client or OpenAICompatibleChatClient()
    response_text = client.create_chat_completion(prompt)
    return parse_grounded_response(response_text=response_text, selected_sentences=selected_sentences, prompt=prompt)


def generate_subqueries(
    question: str,
    context_titles: List[str],
    client: OpenAICompatibleChatClient | None = None,
) -> List[str]:
    """Generate up to two structured follow-up retrieval queries; fail closed on errors."""
    if not question.strip():
        return []

    client = client or OpenAICompatibleChatClient()
    system_prompt = "You solve multi-hop QA by identifying missing information."
    user_prompt = build_subquery_prompt(question=question, context_titles=context_titles)

    try:
        response_text = client.create_structured_chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        payload = json.loads(response_text)
        raw_subqueries = payload.get("subqueries", [])
        if not isinstance(raw_subqueries, list):
            return []

        cleaned: List[str] = []
        for item in raw_subqueries[:2]:
            if not isinstance(item, str):
                continue
            candidate = " ".join(item.split()).strip()
            if not candidate:
                continue
            cleaned.append(candidate)
        return cleaned
    except (LLMConfigurationError, LLMRequestError, json.JSONDecodeError):
        return []
