"""LLM 분석 결과의 구조화 출력 스키마.

각 분석 항목은 점수(0~100)와 근거(원문 발췌 + 코멘트)를 반환한다.
"""
from pydantic import BaseModel, Field


class Evidence(BaseModel):
    """분석 근거: 스크립트 원문 발췌와 LLM 코멘트."""

    excerpt: str = Field(description="스크립트 원문 발췌")
    comment: str = Field(description="해당 발췌에 대한 분석 코멘트")
    timestamp: str | None = Field(default=None, description="발화 시각 (가능한 경우)")


class CriterionResult(BaseModel):
    """단일 분석 항목 결과."""

    criterion: str = Field(description="분석 항목 키 (예: repetition, clarity)")
    score: float = Field(ge=0, le=100, description="0~100 점수")
    summary: str = Field(description="항목 요약")
    evidences: list[Evidence] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list, description="개선 제안")


class LectureAnalysis(BaseModel):
    """단일 강의 분석 결과 집합."""

    lecture_id: str
    instructor_id: str
    criteria: list[CriterionResult]
    overall_score: float = Field(ge=0, le=100)
