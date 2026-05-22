"""Pydantic models for /jdfit LLM tool-use I/O."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ExtractedRequirement(BaseModel):
    spec: str
    category: Literal["must_have", "nice_to_have", "soft"]


class ExtractionResult(BaseModel):
    requirements: list[ExtractedRequirement]


class ScoredRequirement(BaseModel):
    spec: str
    category: Literal["must_have", "nice_to_have", "soft"]
    score: int = Field(ge=0, le=10)
    reasoning: str


class JdfitReport(BaseModel):
    requirements: list[ScoredRequirement]
    overall_score: int = Field(ge=0, le=10)
    strengths: list[str]
    gaps: list[str] = Field(default_factory=list)
    summary: str
