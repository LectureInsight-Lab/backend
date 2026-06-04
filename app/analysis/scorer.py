"""항목별 점수 → 가중치 → 종합 점수 계산.

가중치는 configs/analysis_criteria.yaml 에서 관리.
"""
from app.analysis.schemas import CriterionResult, LectureAnalysis


def aggregate(
    lecture_id: str,
    instructor_id: str,
    results: list[CriterionResult],
    weights: dict[str, float] | None = None,
) -> LectureAnalysis:
    """가중 평균으로 종합 점수 계산 (placeholder)."""
    raise NotImplementedError
