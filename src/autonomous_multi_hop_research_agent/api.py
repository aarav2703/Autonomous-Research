"""FastAPI backend for the autonomous research workflow."""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch

from autonomous_multi_hop_research_agent.workflow import AutonomousResearchWorkflow


if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

load_dotenv()


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    retrieval_top_k: int = Field(default=5, ge=0)
    evidence_top_k: int = Field(default=5, ge=0)
    use_hybrid_retrieval: bool = Field(default=True)
    use_multi_hop: bool = Field(default=False)


class EvidenceItem(BaseModel):
    sentence_id: str
    title: str
    sentence_index: int
    sentence_text: str


class AskResponse(BaseModel):
    answer: str
    reasoning: list[str]
    evidence: list[EvidenceItem]
    status: str
    metadata: dict[str, str]
    retrieval_debug: dict[str, object]
    execution_trace: list[str]


class HealthResponse(BaseModel):
    status: str


@lru_cache(maxsize=1)
def get_workflow() -> AutonomousResearchWorkflow:
    """Create and cache the production workflow instance."""
    return AutonomousResearchWorkflow()


def create_app() -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(title="Autonomous Multi-hop Research Agent API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5173",
            "http://localhost:5173",
            "http://127.0.0.1:3000",
            "http://localhost:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.post("/ask", response_model=AskResponse)
    def ask(payload: AskRequest) -> AskResponse:
        workflow = get_workflow()
        result = workflow.run(
            question=payload.question,
            retrieval_top_k=payload.retrieval_top_k,
            evidence_top_k=payload.evidence_top_k,
            use_hybrid_retrieval=payload.use_hybrid_retrieval,
            use_multi_hop=payload.use_multi_hop,
        )
        response = result["response"]
        return AskResponse(
            answer=response["answer"],
            reasoning=response["reasoning"],
            evidence=[EvidenceItem(**item) for item in response["evidence"]],
            status=response["status"],
            metadata=response["metadata"],
            retrieval_debug=response.get("retrieval_debug", {}),
            execution_trace=response["execution_trace"],
        )

    return app


app = create_app()
