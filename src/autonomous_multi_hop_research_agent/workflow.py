"""LangGraph orchestration for the multi-hop research pipeline."""

from __future__ import annotations

from typing import Any, TypedDict

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
)
from autonomous_multi_hop_research_agent.retrieval import DenseRetriever


class ResearchGraphState(TypedDict, total=False):
    question_id: str
    question: str
    normalized_question: str
    retrieval_top_k: int
    evidence_top_k: int
    use_multi_hop: bool
    use_hybrid_retrieval: bool
    retrieved_chunks: list[dict[str, Any]]
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

    def multi_hop_retrieval_node(self, state: ResearchGraphState) -> ResearchGraphState:
        top_k = state.get("retrieval_top_k", 5)
        use_multi_hop = state.get("use_multi_hop", self.use_multi_hop)
        use_hybrid_retrieval = state.get("use_hybrid_retrieval", self.use_hybrid_retrieval)
        if not state.get("normalized_question", "").strip():
            return {
                "retrieved_chunks": [],
                "retrieval_debug": {},
                "execution_trace": self._append_trace(
                    state,
                    "multi_hop_retrieval: skipped because the normalized question is empty; returning no chunks",
                ),
            }
        if top_k <= 0:
            return {
                "retrieved_chunks": [],
                "retrieval_debug": {},
                "execution_trace": self._append_trace(
                    state,
                    f"multi_hop_retrieval: skipped because retrieval_top_k={top_k} is not valid; returning no chunks",
                ),
            }

        if not use_multi_hop:
            active_retriever = self.hybrid_retriever if use_hybrid_retrieval else self.retriever
            mode_label = "hybrid" if use_hybrid_retrieval else "single-hop dense"
            if use_hybrid_retrieval:
                hybrid_result = self.hybrid_retriever.retrieve_with_debug(state["normalized_question"], top_k=top_k)
                retrieved = hybrid_result.merged_chunks
                dense_titles = hybrid_result.dense_chunks["title"].tolist()
                bm25_titles = hybrid_result.bm25_chunks["title"].tolist()
                merged_titles = retrieved["title"].tolist()
                title_boosts = hybrid_result.title_boosts
            else:
                retrieved = active_retriever.retrieve(state["normalized_question"], top_k=top_k)
                dense_titles = retrieved["title"].tolist()
                bm25_titles = []
                merged_titles = retrieved["title"].tolist()
                title_boosts = []
            trace = self._append_trace(
                state,
                (
                    f"multi_hop_retrieval: {mode_label} fallback retrieved {len(retrieved)} paragraph chunks "
                    f"with top_k={top_k}"
                    if not retrieved.empty
                    else f"multi_hop_retrieval: {mode_label} fallback returned no chunks for top_k={top_k}"
                ),
            )
            return {
                "retrieved_chunks": retrieved.to_dict(orient="records"),
                "retrieval_debug": {
                    "mode": "hybrid" if use_hybrid_retrieval else "single_hop_dense",
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

        active_multi_hop_retriever = self.hybrid_multi_hop_retriever if use_hybrid_retrieval else self.multi_hop_retriever
        retrieval_result = active_multi_hop_retriever.retrieve_with_debug(
            question=state["normalized_question"],
            top_k=top_k,
        )
        hop1_titles = retrieval_result.hop1_chunks["title"].tolist() if not retrieval_result.hop1_chunks.empty else []
        hop2_titles = retrieval_result.hop2_chunks["title"].tolist() if not retrieval_result.hop2_chunks.empty else []
        merged_titles = retrieval_result.merged_chunks["title"].tolist() if not retrieval_result.merged_chunks.empty else []
        trace = self._append_trace(
            state,
            (
                f"multi_hop_retrieval: merged {len(retrieval_result.merged_chunks)} paragraph chunks "
                f"from hop1={len(retrieval_result.hop1_chunks)} and hop2={len(retrieval_result.hop2_chunks)}"
                if not retrieval_result.merged_chunks.empty
                else "multi_hop_retrieval: no chunks returned after both hops"
            ),
        )
        trace = self._append_trace(
            {"execution_trace": trace},
            f"multi_hop_retrieval: hop1 titles={hop1_titles[:5]}",
        )
        if retrieval_result.extracted_entities:
            trace = self._append_trace(
                {"execution_trace": trace},
                f"multi_hop_retrieval: extracted entities={retrieval_result.extracted_entities}",
            )
            trace = self._append_trace(
                {"execution_trace": trace},
                f"multi_hop_retrieval: hop2 queries={retrieval_result.hop2_queries}",
            )
            trace = self._append_trace(
                {"execution_trace": trace},
                f"multi_hop_retrieval: hop2 titles={hop2_titles[:10]}",
            )
        elif retrieval_result.fallback_reason:
            trace = self._append_trace(
                {"execution_trace": trace},
                f"multi_hop_retrieval: {retrieval_result.fallback_reason}",
            )
        trace = self._append_trace(
            {"execution_trace": trace},
            f"multi_hop_retrieval: merged titles={merged_titles[:10]}",
        )
        return {
            "retrieved_chunks": retrieval_result.merged_chunks.to_dict(orient="records"),
            "retrieval_debug": {
                "mode": "multi_hop",
                "base_retriever": "hybrid" if use_hybrid_retrieval else "dense",
                "hop1_titles": hop1_titles,
                "entities": retrieval_result.extracted_entities,
                "hop2_queries": retrieval_result.hop2_queries,
                "hop2_titles": hop2_titles,
                "merged_titles": merged_titles,
                "fallback_reason": retrieval_result.fallback_reason,
            },
            "execution_trace": trace,
        }

    def evidence_selection_node(self, state: ResearchGraphState) -> ResearchGraphState:
        import pandas as pd

        top_k = state.get("evidence_top_k", 5)
        if top_k <= 0:
            return {
                "selected_evidence": [],
                "execution_trace": self._append_trace(
                    state,
                    f"evidence_selection: skipped because evidence_top_k={top_k} is not valid",
                ),
            }

        retrieved_df = pd.DataFrame(state.get("retrieved_chunks", []))
        if retrieved_df.empty:
            return {
                "selected_evidence": [],
                "execution_trace": self._append_trace(
                    state,
                    "evidence_selection: skipped because retrieval returned no chunks",
                ),
            }

        evidence: EvidenceSelectionResult = self.evidence_selector.select_evidence(
            question=state["normalized_question"],
            retrieved_chunks=retrieved_df,
            question_id=state.get("question_id", ""),
            top_k_sentences=top_k,
        )
        return {
            "selected_evidence": evidence.selected_sentences.to_dict(orient="records"),
            "execution_trace": self._append_trace(
                state,
                "evidence_selection: "
                f"selected {len(evidence.selected_sentences)} sentences, "
                f"supporting_fact_recall={evidence.supporting_fact_recall:.2f}",
            ),
        }

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
        response = {
            "question": state["question"],
            "normalized_question": state["normalized_question"],
            "answer": grounded["answer"],
            "reasoning": grounded["reasoning_trace"],
            "evidence": grounded["cited_evidence"],
            "prompt": grounded["prompt"],
            "status": grounded["status"],
            "metadata": {
                "failure_reason": grounded["failure_reason"],
                "use_multi_hop": str(state.get("use_multi_hop", self.use_multi_hop)),
                "use_hybrid_retrieval": str(state.get("use_hybrid_retrieval", self.use_hybrid_retrieval)),
            },
            "retrieval_debug": state.get("retrieval_debug", {}),
            "execution_trace": updated_trace,
        }
        return {
            "response": response,
            "execution_trace": updated_trace,
        }

    def _build_graph(self) -> StateGraph:
        graph_builder = StateGraph(ResearchGraphState)
        graph_builder.add_node("question_normalization", self.question_normalization_node)
        graph_builder.add_node("multi_hop_retrieval", self.multi_hop_retrieval_node)
        graph_builder.add_node("evidence_selection", self.evidence_selection_node)
        graph_builder.add_node("answer_generation", self.answer_generation_node)
        graph_builder.add_node("response_formatting", self.response_formatting_node)

        graph_builder.add_edge(START, "question_normalization")
        graph_builder.add_edge("question_normalization", "multi_hop_retrieval")
        graph_builder.add_edge("multi_hop_retrieval", "evidence_selection")
        graph_builder.add_edge("evidence_selection", "answer_generation")
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
            "use_multi_hop": self.use_multi_hop if use_multi_hop is None else use_multi_hop,
            "use_hybrid_retrieval": (
                self.use_hybrid_retrieval if use_hybrid_retrieval is None else use_hybrid_retrieval
            ),
            "execution_trace": [],
        }
        return self.graph.invoke(initial_state)
