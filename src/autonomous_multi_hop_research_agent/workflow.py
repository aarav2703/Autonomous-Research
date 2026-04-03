"""LangGraph orchestration for the multi-hop research pipeline."""

from __future__ import annotations

import logging
from typing import Any, TypedDict

import numpy as np
from langgraph.graph import END, START, StateGraph

from autonomous_multi_hop_research_agent.evidence import EvidenceSelectionResult, EvidenceSelector
from autonomous_multi_hop_research_agent.hybrid_retrieval import HybridRetriever
from autonomous_multi_hop_research_agent.multi_hop_retrieval import MultiHopRetriever
from autonomous_multi_hop_research_agent.rag import (
    CitationValidationError,
    GroundedAnswer,
    LLMConfigurationError,
    LLMRequestError,
    build_safe_failure_answer,
    generate_grounded_answer,
    generate_subqueries,
)
from autonomous_multi_hop_research_agent.retrieval import DenseRetriever


logger = logging.getLogger(__name__)


class ResearchGraphState(TypedDict, total=False):
    question_id: str
    question: str
    normalized_question: str
    retrieval_top_k: int
    evidence_top_k: int
    use_multi_hop: bool
    use_hybrid_retrieval: bool
    retrieval_confidence: float
    evidence_confidence: float
    retrieval_mode: str
    hop_count: int
    max_hops: int
    policy_next: str
    evidence_attempted: bool
    tried_modes: list[str]
    subqueries: list[str]
    planner_decision: str
    planner_enabled: bool
    retrieval_recall_estimate: float
    evidence_recall_estimate: float
    used_multi_hop: bool
    used_subqueries: bool
    reranked_candidate_count: int
    retrieved_chunks: list[dict[str, Any]]
    stage_counts: dict[str, int]
    retrieval_debug: dict[str, Any]
    selected_evidence: list[dict[str, Any]]
    grounded_answer: dict[str, Any]
    response: dict[str, Any]
    execution_trace: list[str]


class AutonomousResearchWorkflow:
    """Structured LangGraph workflow for retrieval, evidence, and grounded answering."""

    def __init__(
        self,
        retriever: DenseRetriever | None = None,
        hybrid_retriever: HybridRetriever | None = None,
        multi_hop_retriever: MultiHopRetriever | None = None,
        hybrid_multi_hop_retriever: MultiHopRetriever | None = None,
        evidence_selector: EvidenceSelector | None = None,
        use_multi_hop: bool = True,
        use_hybrid_retrieval: bool = False,
        enable_planner: bool = True,
    ) -> None:
        self.retriever = retriever or DenseRetriever()
        self.hybrid_retriever = hybrid_retriever or HybridRetriever(base_retriever=self.retriever)
        self.multi_hop_retriever = multi_hop_retriever or MultiHopRetriever(base_retriever=self.retriever)
        self.hybrid_multi_hop_retriever = hybrid_multi_hop_retriever or MultiHopRetriever(
            base_retriever=self.hybrid_retriever
        )
        self.evidence_selector = evidence_selector or EvidenceSelector()
        self.use_multi_hop = use_multi_hop
        self.use_hybrid_retrieval = use_hybrid_retrieval
        self.enable_planner = enable_planner
        self.graph = self._build_graph().compile()

    def _append_trace(self, state: ResearchGraphState, message: str) -> list[str]:
        trace = list(state.get("execution_trace", []))
        trace.append(message)
        return trace

    def question_normalization_node(self, state: ResearchGraphState) -> ResearchGraphState:
        normalized_question = " ".join(state["question"].split())
        return {
            "normalized_question": normalized_question,
            "execution_trace": self._append_trace(
                state,
                f"question_normalization: normalized question to '{normalized_question}'",
            ),
        }

    def _clamp_confidence(self, value: float) -> float:
        return float(max(0.0, min(1.0, value)))

    def _run_retrieval_mode(self, state: ResearchGraphState, mode: str) -> ResearchGraphState:
        import pandas as pd

        top_k = state.get("retrieval_top_k", 5)
        if not state.get("normalized_question", "").strip():
            return {
                "retrieved_chunks": [],
                "retrieval_debug": {},
                "retrieval_mode": mode,
                "selected_evidence": [],
                "execution_trace": self._append_trace(
                    state,
                    f"{mode}_retrieval: skipped because the normalized question is empty; returning no chunks",
                ),
            }
        if top_k <= 0:
            return {
                "retrieved_chunks": [],
                "retrieval_debug": {},
                "retrieval_mode": mode,
                "selected_evidence": [],
                "execution_trace": self._append_trace(
                    state,
                    f"{mode}_retrieval: skipped because retrieval_top_k={top_k} is not valid; returning no chunks",
                ),
            }

        if mode != "multi-hop":
            if mode == "hybrid":
                hybrid_result = self.hybrid_retriever.retrieve_with_debug(state["normalized_question"], top_k=top_k)
                retrieved = hybrid_result.merged_chunks
                dense_titles = hybrid_result.dense_chunks["title"].tolist()
                bm25_titles = hybrid_result.bm25_chunks["title"].tolist()
                merged_titles = retrieved["title"].tolist()
                title_boosts = hybrid_result.title_boosts
            else:
                retrieved = self.retriever.retrieve(state["normalized_question"], top_k=top_k)
                dense_titles = retrieved["title"].tolist()
                bm25_titles = []
                merged_titles = retrieved["title"].tolist()
                title_boosts = []
            trace = self._append_trace(
                state,
                (
                    f"{mode}_retrieval: retrieved {len(retrieved)} paragraph chunks "
                    f"with top_k={top_k}"
                    if not retrieved.empty
                    else f"{mode}_retrieval: returned no chunks for top_k={top_k}"
                ),
            )
            return {
                "retrieved_chunks": retrieved.to_dict(orient="records"),
                "retrieval_mode": mode,
                "selected_evidence": [],
                "retrieval_debug": {
                    "mode": "hybrid" if mode == "hybrid" else "single_hop_dense",
                    "dense_titles": dense_titles,
                    "bm25_titles": bm25_titles,
                    "title_boosts": title_boosts,
                    "hop1_titles": retrieved["title"].tolist(),
                    "entities": [],
                    "hop2_queries": [],
                    "hop2_titles": [],
                    "merged_titles": merged_titles,
                },
                "execution_trace": trace,
            }

        use_hybrid_base = state.get("use_hybrid_retrieval", self.use_hybrid_retrieval) or state.get(
            "retrieval_mode", ""
        ) == "hybrid"
        active_multi_hop_retriever = self.hybrid_multi_hop_retriever if use_hybrid_base else self.multi_hop_retriever
        retrieval_result = active_multi_hop_retriever.retrieve_with_debug(
            question=state["normalized_question"],
            top_k=top_k,
        )
        merged_chunks = retrieval_result.merged_chunks.copy()
        subqueries = [query for query in state.get("subqueries", []) if str(query).strip()]
        subquery_titles: dict[str, list[str]] = {}
        first_hop_titles = set(retrieval_result.hop1_chunks["title"].astype(str).tolist()) if not retrieval_result.hop1_chunks.empty else set()
        if subqueries:
            subquery_frames: list[pd.DataFrame] = [merged_chunks]
            for subquery in subqueries:
                subquery_frame = active_multi_hop_retriever.base_retriever.retrieve(subquery, top_k=top_k)
                subquery_titles[subquery] = subquery_frame["title"].tolist()[:top_k] if not subquery_frame.empty else []
                if not subquery_frame.empty:
                    subquery_frame = subquery_frame.copy()
                    subquery_frame["score"] = subquery_frame["score"].astype(float)
                    subquery_frame.loc[~subquery_frame["title"].astype(str).isin(first_hop_titles), "score"] += 0.1
                subquery_frames.append(subquery_frame)
            merged_chunks = (
                pd.concat(subquery_frames, ignore_index=True)
                .sort_values("score", ascending=False)
                .drop_duplicates(subset=["title"], keep="first")
                .reset_index(drop=True)
            )
            new_title_mask = ~merged_chunks["title"].astype(str).isin(first_hop_titles)
            new_title_count = int(new_title_mask.sum())
            if new_title_count > 0:
                retained_new_titles = merged_chunks.loc[new_title_mask].head(2)
                retained_old_titles = merged_chunks.loc[~new_title_mask].head(max(0, top_k - len(retained_new_titles)))
                merged_chunks = (
                    pd.concat([retained_new_titles, retained_old_titles], ignore_index=True)
                    .drop_duplicates(subset=["title"], keep="first")
                    .head(top_k)
                    .reset_index(drop=True)
                )
            else:
                merged_chunks = merged_chunks.head(top_k).reset_index(drop=True)

        hop1_titles = retrieval_result.hop1_chunks["title"].tolist() if not retrieval_result.hop1_chunks.empty else []
        hop2_titles = retrieval_result.hop2_chunks["title"].tolist() if not retrieval_result.hop2_chunks.empty else []
        merged_titles = merged_chunks["title"].tolist() if not merged_chunks.empty else []
        trace = self._append_trace(
            state,
            (
                f"multi-hop_retrieval: merged {len(merged_chunks)} paragraph chunks "
                f"from hop1={len(retrieval_result.hop1_chunks)} and hop2={len(retrieval_result.hop2_chunks)}"
                if not merged_chunks.empty
                else "multi-hop_retrieval: no chunks returned after both hops"
            ),
        )
        trace = self._append_trace(
            {"execution_trace": trace},
            f"multi-hop_retrieval: hop1 titles={hop1_titles[:5]}",
        )
        if retrieval_result.extracted_entities:
            trace = self._append_trace(
                {"execution_trace": trace},
                f"multi-hop_retrieval: extracted entities={retrieval_result.extracted_entities}",
            )
            trace = self._append_trace(
                {"execution_trace": trace},
                f"multi-hop_retrieval: hop2 queries={retrieval_result.hop2_queries}",
            )
            trace = self._append_trace(
                {"execution_trace": trace},
                f"multi-hop_retrieval: hop2 titles={hop2_titles[:10]}",
            )
        elif retrieval_result.fallback_reason:
            trace = self._append_trace(
                {"execution_trace": trace},
                f"multi-hop_retrieval: {retrieval_result.fallback_reason}",
            )
        trace = self._append_trace(
            {"execution_trace": trace},
            f"multi-hop_retrieval: merged titles={merged_titles[:10]}",
        )
        if subqueries:
            trace = self._append_trace(
                {"execution_trace": trace},
                f"multi-hop_retrieval: subquery retrieval used subqueries={subqueries}",
            )
            trace = self._append_trace(
                {"execution_trace": trace},
                f"multi-hop_retrieval: hop2 added {len([title for title in merged_titles if title not in first_hop_titles])} new titles",
            )
        used_multi_hop = bool(retrieval_result.hop2_chunks.shape[0] > 0 or retrieval_result.extracted_entities)
        used_subqueries = bool(subqueries)
        return {
            "retrieved_chunks": merged_chunks.to_dict(orient="records"),
            "retrieval_mode": "multi-hop",
            "selected_evidence": [],
            "used_multi_hop": used_multi_hop,
            "used_subqueries": used_subqueries,
            "retrieval_debug": {
                "mode": "multi_hop",
                "base_retriever": "hybrid" if use_hybrid_base else "dense",
                "hop1_titles": hop1_titles,
                "entities": retrieval_result.extracted_entities,
                "hop2_queries": retrieval_result.hop2_queries,
                "hop2_titles": hop2_titles,
                "merged_titles": merged_titles,
                "subqueries": subqueries,
                "subquery_titles": subquery_titles,
                "fallback_reason": retrieval_result.fallback_reason,
            },
            "execution_trace": trace,
        }

    def dense_retrieval_node(self, state: ResearchGraphState) -> ResearchGraphState:
        return self._run_retrieval_mode(state=state, mode="dense")

    def hybrid_retrieval_node(self, state: ResearchGraphState) -> ResearchGraphState:
        return self._run_retrieval_mode(state=state, mode="hybrid")

    def multi_hop_retrieval_node(self, state: ResearchGraphState) -> ResearchGraphState:
        return self._run_retrieval_mode(state=state, mode="multi-hop")

    def evidence_selection_node(self, state: ResearchGraphState) -> ResearchGraphState:
        import pandas as pd

        top_k = state.get("evidence_top_k", 5)
        if top_k <= 0:
            return {
                "selected_evidence": [],
                "evidence_attempted": True,
                "execution_trace": self._append_trace(
                    state,
                    f"evidence_selection: skipped because evidence_top_k={top_k} is not valid",
                ),
            }

        retrieved_df = pd.DataFrame(state.get("retrieved_chunks", []))
        if retrieved_df.empty:
            return {
                "selected_evidence": [],
                "evidence_attempted": True,
                "reranked_candidate_count": 0,
                "execution_trace": self._append_trace(
                    state,
                    "evidence_selection: skipped because retrieval returned no chunks",
                ),
            }

        reranked_candidates = self.evidence_selector.candidate_sentences_for_retrieved(retrieved_df)
        reranked_candidate_count = int(len(reranked_candidates))

        evidence: EvidenceSelectionResult = self.evidence_selector.select_evidence(
            question=state["normalized_question"],
            retrieved_chunks=retrieved_df,
            question_id=state.get("question_id", ""),
            top_k_sentences=top_k,
        )
        return {
            "selected_evidence": evidence.selected_sentences.to_dict(orient="records"),
            "evidence_attempted": True,
            "reranked_candidate_count": reranked_candidate_count,
            "execution_trace": self._append_trace(
                state,
                "evidence_selection: "
                f"reranked {reranked_candidate_count} candidates, "
                f"selected {len(evidence.selected_sentences)} sentences, "
                f"supporting_fact_recall={evidence.supporting_fact_recall:.2f}",
            ),
        }

    def compute_confidence_node(self, state: ResearchGraphState) -> ResearchGraphState:
        import pandas as pd

        retrieved_df = pd.DataFrame(state.get("retrieved_chunks", []))
        evidence_df = pd.DataFrame(state.get("selected_evidence", []))
        retrieval_top_k = max(1, int(state.get("retrieval_top_k", 5)))

        retrieval_confidence = 0.0
        if not retrieved_df.empty and "score" in retrieved_df.columns:
            top_scores = retrieved_df["score"].astype(float).head(retrieval_top_k).to_numpy(dtype=float)
            if top_scores.size > 0:
                score_mean = float(np.mean(top_scores))
                score_std = float(np.std(top_scores))
                retrieval_confidence = self._clamp_confidence(score_mean - score_std)

        evidence_confidence = 0.0
        if not evidence_df.empty:
            if "final_score" in evidence_df.columns:
                evidence_scores = evidence_df["final_score"].astype(float).to_numpy(dtype=float)
            elif "evidence_score" in evidence_df.columns:
                evidence_scores = evidence_df["evidence_score"].astype(float).to_numpy(dtype=float)
            else:
                evidence_scores = np.asarray([], dtype=float)
            if evidence_scores.size > 0:
                evidence_mean = float(np.mean(evidence_scores))
                evidence_std = float(np.std(evidence_scores))
                evidence_confidence = self._clamp_confidence(evidence_mean - evidence_std)

        retrieval_titles = retrieved_df["title"].astype(str) if not retrieved_df.empty and "title" in retrieved_df.columns else pd.Series(dtype=str)
        evidence_titles = evidence_df["title"].astype(str) if not evidence_df.empty and "title" in evidence_df.columns else pd.Series(dtype=str)
        unique_retrieval_titles = int(retrieval_titles.nunique()) if not retrieval_titles.empty else 0
        unique_evidence_titles = int(evidence_titles.nunique()) if not evidence_titles.empty else 0
        retrieval_recall_estimate = self._clamp_confidence(unique_retrieval_titles / max(1, retrieval_top_k))
        evidence_recall_estimate = self._clamp_confidence(
            unique_evidence_titles / max(1, len(evidence_df)) if len(evidence_df) > 0 else 0.0
        )
        used_multi_hop = bool(state.get("retrieval_mode") == "multi-hop" or state.get("used_multi_hop", False))
        used_subqueries = bool(state.get("subqueries", [])) or bool(state.get("used_subqueries", False))

        logger.info(
            "Confidence values | mode=%s retrieval_confidence=%.3f evidence_confidence=%.3f retrieval_recall_estimate=%.3f evidence_recall_estimate=%.3f",
            state.get("retrieval_mode", "unknown"),
            retrieval_confidence,
            evidence_confidence,
            retrieval_recall_estimate,
            evidence_recall_estimate,
        )

        trace = self._append_trace(
            state,
            "compute_confidence: "
            f"mode={state.get('retrieval_mode', 'unknown')} | "
            f"retrieval_confidence={retrieval_confidence:.3f} | "
            f"evidence_confidence={evidence_confidence:.3f}",
        )
        return {
            "retrieval_confidence": retrieval_confidence,
            "evidence_confidence": evidence_confidence,
            "retrieval_recall_estimate": retrieval_recall_estimate,
            "evidence_recall_estimate": evidence_recall_estimate,
            "used_multi_hop": used_multi_hop,
            "used_subqueries": used_subqueries,
            "execution_trace": trace,
        }

    def planner_node(self, state: ResearchGraphState) -> ResearchGraphState:
        planner_enabled = bool(state.get("planner_enabled", self.enable_planner))
        hop_count = int(state.get("hop_count", 0))
        max_hops = int(state.get("max_hops", 2))
        evidence_confidence = float(state.get("evidence_confidence", 0.0))
        retrieval_confidence = float(state.get("retrieval_confidence", 0.0))

        if not planner_enabled:
            trace = self._append_trace(
                state,
                "planner: disabled, decision=STOP",
            )
            return {
                "planner_decision": "STOP",
                "execution_trace": trace,
            }

        if evidence_confidence > 0.5:
            decision = "STOP"
        elif retrieval_confidence < 0.3:
            decision = "RETRIEVE_MORE"
        elif hop_count == 0:
            decision = "RETRIEVE_MORE"
        else:
            decision = "STOP"

        updated_hop_count = hop_count + 1 if decision == "RETRIEVE_MORE" else hop_count
        trace = self._append_trace(
            state,
            "planner: "
            f"retrieval_confidence={retrieval_confidence:.3f} | evidence_confidence={evidence_confidence:.3f} | hop_count={hop_count}/{max_hops} | decision={decision}",
        )
        return {
            "planner_decision": decision,
            "hop_count": updated_hop_count,
            "execution_trace": trace,
        }

    def subquery_generation_node(self, state: ResearchGraphState) -> ResearchGraphState:
        retrieved_chunks = state.get("retrieved_chunks", [])
        context_titles = [str(item.get("title", "")).strip() for item in retrieved_chunks if item.get("title")]
        unique_titles = list(dict.fromkeys(title for title in context_titles if title))
        subqueries = generate_subqueries(
            question=state.get("normalized_question", state.get("question", "")),
            context_titles=unique_titles,
        )

        if not subqueries:
            trace = self._append_trace(
                state,
                "subquery_generation: no subqueries generated; fallback to answer_generation",
            )
            return {
                "subqueries": [],
                "planner_decision": "STOP",
                "execution_trace": trace,
            }

        trace = self._append_trace(
            state,
            f"subquery_generation: generated subqueries={subqueries}",
        )
        return {
            "subqueries": subqueries,
            "planner_decision": "RETRIEVE_MORE",
            "execution_trace": trace,
        }

    def _route_from_planner(self, state: ResearchGraphState) -> str:
        if state.get("planner_decision") == "RETRIEVE_MORE":
            return "subquery_generation"
        return "answer_generation"

    def _route_from_subquery_generation(self, state: ResearchGraphState) -> str:
        if state.get("subqueries"):
            return "multi_hop_retrieval"
        return "answer_generation"

    def _route_from_question_normalization(self, state: ResearchGraphState) -> str:
        if state.get("use_multi_hop", self.use_multi_hop):
            return "multi_hop_retrieval"
        if state.get("use_hybrid_retrieval", self.use_hybrid_retrieval):
            return "hybrid_retrieval"
        return "dense_retrieval"

    def retrieval_policy_node(self, state: ResearchGraphState) -> ResearchGraphState:
        hop_count = int(state.get("hop_count", 0))
        max_hops = int(state.get("max_hops", 2))
        retrieval_mode = state.get("retrieval_mode", "dense")
        retrieval_confidence = float(state.get("retrieval_confidence", 0.0))
        evidence_confidence = float(state.get("evidence_confidence", 0.0))
        tried_modes = list(state.get("tried_modes", []))
        if retrieval_mode and retrieval_mode not in tried_modes:
            tried_modes.append(retrieval_mode)

        forced_multi_hop = state.get("use_multi_hop", self.use_multi_hop)
        forced_hybrid = state.get("use_hybrid_retrieval", self.use_hybrid_retrieval)
        evidence_attempted = bool(state.get("evidence_attempted", False))
        has_evidence = bool(state.get("selected_evidence", []))

        if hop_count >= max_hops:
            trace = self._append_trace(
                state,
                "retrieval_policy: max_hops reached; forcing answer_generation",
            )
            return {
                "policy_next": "answer_generation",
                "hop_count": hop_count,
                "retrieval_mode": retrieval_mode,
                "tried_modes": tried_modes,
                "execution_trace": trace,
            }

        next_action = "proceed"
        if hop_count >= max_hops:
            next_action = "proceed"
        elif forced_multi_hop and retrieval_mode != "multi-hop" and "multi-hop" not in tried_modes:
            next_action = "to_multi_hop"
        elif forced_hybrid and retrieval_mode == "dense" and "hybrid" not in tried_modes:
            next_action = "to_hybrid"
        elif retrieval_confidence < 0.4 and retrieval_mode == "dense" and "hybrid" not in tried_modes:
            next_action = "to_hybrid"
        elif evidence_confidence < 0.3 and retrieval_mode != "multi-hop" and "multi-hop" not in tried_modes:
            next_action = "to_multi_hop"

        if next_action == "to_hybrid":
            policy_next = "hybrid_retrieval"
            hop_count += 1
            chosen_mode = "hybrid"
        elif next_action == "to_multi_hop":
            policy_next = "multi_hop_retrieval"
            hop_count += 1
            chosen_mode = "multi-hop"
        else:
            if has_evidence or evidence_attempted or hop_count >= max_hops:
                policy_next = "answer_generation"
            else:
                policy_next = "evidence_selection"
            chosen_mode = retrieval_mode

        trace = self._append_trace(
            state,
            "retrieval_policy: "
            f"mode={retrieval_mode} | retrieval_confidence={retrieval_confidence:.3f} | "
            f"evidence_confidence={evidence_confidence:.3f} | decision={policy_next}",
        )
        return {
            "policy_next": policy_next,
            "hop_count": hop_count,
            "retrieval_mode": chosen_mode,
            "tried_modes": tried_modes,
            "execution_trace": trace,
        }

    def _route_from_policy(self, state: ResearchGraphState) -> str:
        next_node = state.get("policy_next", "answer_generation")
        if next_node in {"hybrid_retrieval", "multi_hop_retrieval", "evidence_selection", "answer_generation"}:
            return next_node
        return "answer_generation"

    def answer_generation_node(self, state: ResearchGraphState) -> ResearchGraphState:
        import pandas as pd

        retrieved_df = pd.DataFrame(state.get("retrieved_chunks", []))
        evidence_df = pd.DataFrame(state.get("selected_evidence", []))
        try:
            grounded: GroundedAnswer = generate_grounded_answer(
                question=state["normalized_question"],
                retrieved_chunks=retrieved_df,
                selected_sentences=evidence_df,
            )
        except CitationValidationError as exc:
            grounded = build_safe_failure_answer(reason=f"Citation validation failed: {exc}")
        except (LLMConfigurationError, LLMRequestError) as exc:
            grounded = build_safe_failure_answer(reason=f"Answer generation failed closed: {exc}")

        return {
            "grounded_answer": {
                "answer": grounded.answer,
                "reasoning_trace": grounded.reasoning_trace,
                "cited_evidence": grounded.cited_evidence,
                "prompt": grounded.prompt,
                "raw_response_text": grounded.raw_response_text,
                "status": grounded.status,
                "failure_reason": grounded.failure_reason,
            },
            "execution_trace": self._append_trace(
                state,
                "answer_generation: "
                f"generated answer '{grounded.answer}' with {len(grounded.cited_evidence)} citations"
                + (f"; failure_reason={grounded.failure_reason}" if grounded.failure_reason else ""),
            ),
        }

    def response_formatting_node(self, state: ResearchGraphState) -> ResearchGraphState:
        grounded = state["grounded_answer"]
        updated_trace = self._append_trace(
            state,
            "response_formatting: assembled final response payload",
        )
        retrieved_count = len(state.get("retrieved_chunks", []))
        reranked_count = int(state.get("reranked_candidate_count", len(state.get("selected_evidence", []))))
        evidence_count = len(state.get("selected_evidence", []))
        answer_count = 1 if grounded.get("answer") else 0
        stage_counts = {
            "retrieved": retrieved_count,
            "reranked": reranked_count,
            "evidence": evidence_count,
            "answer": answer_count,
            "discarded_after_retrieval": max(0, retrieved_count - reranked_count),
            "discarded_after_rerank": max(0, reranked_count - evidence_count),
            "discarded_after_evidence": max(0, evidence_count - answer_count),
        }
        response = {
            "question": state["question"],
            "normalized_question": state["normalized_question"],
            "answer": grounded["answer"],
            "reasoning": grounded["reasoning_trace"],
            "evidence": grounded["cited_evidence"],
            "retrieved_chunks": state.get("retrieved_chunks", []),
            "selected_evidence": state.get("selected_evidence", []),
            "subqueries": state.get("subqueries", []),
            "hop_count": int(state.get("hop_count", 0)),
            "stage_counts": stage_counts,
            "prompt": grounded["prompt"],
            "status": grounded["status"],
            "metadata": {
                "failure_reason": grounded["failure_reason"],
                "use_multi_hop": str(state.get("use_multi_hop", self.use_multi_hop)),
                "use_hybrid_retrieval": str(state.get("use_hybrid_retrieval", self.use_hybrid_retrieval)),
                "retrieval_mode": str(state.get("retrieval_mode", "dense")),
                "retrieval_confidence": f"{state.get('retrieval_confidence', 0.0):.4f}",
                "evidence_confidence": f"{state.get('evidence_confidence', 0.0):.4f}",
                "retrieval_recall_estimate": f"{state.get('retrieval_recall_estimate', 0.0):.4f}",
                "evidence_recall_estimate": f"{state.get('evidence_recall_estimate', 0.0):.4f}",
                "used_multi_hop": str(state.get("used_multi_hop", False)),
                "used_subqueries": str(state.get("used_subqueries", False)),
            },
            "retrieval_debug": state.get("retrieval_debug", {}),
            "execution_trace": updated_trace,
        }
        return {
            "response": response,
            "execution_trace": updated_trace,
            "stage_counts": stage_counts,
        }

    def _build_graph(self) -> StateGraph:
        graph_builder = StateGraph(ResearchGraphState)
        graph_builder.add_node("question_normalization", self.question_normalization_node)
        graph_builder.add_node("dense_retrieval", self.dense_retrieval_node)
        graph_builder.add_node("hybrid_retrieval", self.hybrid_retrieval_node)
        graph_builder.add_node("multi_hop_retrieval", self.multi_hop_retrieval_node)
        graph_builder.add_node("evidence_selection", self.evidence_selection_node)
        graph_builder.add_node("compute_confidence", self.compute_confidence_node)
        graph_builder.add_node("planner", self.planner_node)
        graph_builder.add_node("subquery_generation", self.subquery_generation_node)
        graph_builder.add_node("answer_generation", self.answer_generation_node)
        graph_builder.add_node("response_formatting", self.response_formatting_node)

        graph_builder.add_edge(START, "question_normalization")
        graph_builder.add_conditional_edges(
            "question_normalization",
            self._route_from_question_normalization,
            {
                "dense_retrieval": "dense_retrieval",
                "hybrid_retrieval": "hybrid_retrieval",
                "multi_hop_retrieval": "multi_hop_retrieval",
            },
        )
        graph_builder.add_edge("dense_retrieval", "evidence_selection")
        graph_builder.add_edge("hybrid_retrieval", "evidence_selection")
        graph_builder.add_edge("multi_hop_retrieval", "evidence_selection")
        graph_builder.add_edge("evidence_selection", "compute_confidence")
        graph_builder.add_edge("compute_confidence", "planner")
        graph_builder.add_conditional_edges(
            "planner",
            self._route_from_planner,
            {
                "subquery_generation": "subquery_generation",
                "answer_generation": "answer_generation",
            },
        )
        graph_builder.add_conditional_edges(
            "subquery_generation",
            self._route_from_subquery_generation,
            {
                "multi_hop_retrieval": "multi_hop_retrieval",
                "answer_generation": "answer_generation",
            },
        )
        graph_builder.add_edge("answer_generation", "response_formatting")
        graph_builder.add_edge("response_formatting", END)
        return graph_builder

    def run(
        self,
        question: str,
        question_id: str = "",
        retrieval_top_k: int = 5,
        evidence_top_k: int = 5,
        use_multi_hop: bool | None = None,
        use_hybrid_retrieval: bool | None = None,
    ) -> ResearchGraphState:
        """Execute the end-to-end graph for one question."""
        initial_state: ResearchGraphState = {
            "question_id": question_id,
            "question": question,
            "retrieval_top_k": retrieval_top_k,
            "evidence_top_k": evidence_top_k,
            "retrieval_confidence": 0.0,
            "evidence_confidence": 0.0,
            "retrieval_recall_estimate": 0.0,
            "evidence_recall_estimate": 0.0,
            "retrieval_mode": "dense",
            "hop_count": 0,
            "max_hops": 2,
            "policy_next": "evidence_selection",
            "evidence_attempted": False,
            "tried_modes": [],
            "subqueries": [],
            "planner_decision": "STOP",
            "planner_enabled": self.enable_planner,
            "used_multi_hop": False,
            "used_subqueries": False,
            "reranked_candidate_count": 0,
            "stage_counts": {},
            "use_multi_hop": self.use_multi_hop if use_multi_hop is None else use_multi_hop,
            "use_hybrid_retrieval": (
                self.use_hybrid_retrieval if use_hybrid_retrieval is None else use_hybrid_retrieval
            ),
            "execution_trace": [],
        }
        return self.graph.invoke(initial_state)
