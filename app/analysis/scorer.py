"""스코어링 (5단계).

18개 항목 점수를 카테고리 평균 → 종합 점수로 집계하고,
다중 강의 분석 시 시계열 트렌드(선형 회귀)를 계산한다.

카테고리 가중치는 ``configs/checklist.yaml`` 의 ``category_weights`` 사용:
    structure   25%
    concept     25%
    practice    20%
    language    15%
    interaction 15%

트렌드 라벨:
    slope >= +0.05  → "improving"
    slope <= -0.05  → "declining"
    else            → "stable"

주차 집계:
    1주차(02-02 ~ 02-06), 2주차(02-09 ~ 02-13), 3주차(02-23 ~ 02-27)
"""
from app.analysis.schemas import (
    CategoryScore,
    InstructorScorecard,
    ItemScore,
    TrendPoint,
)
from app.core.checklist import Checklist


def category_scores(items: list[ItemScore], checklist: Checklist) -> list[CategoryScore]:
    """카테고리별 평균 점수 + 가중치 부착 (placeholder)."""
    raise NotImplementedError


def overall_score(categories: list[CategoryScore]) -> float:
    """카테고리별 가중 평균 → 종합 점수 (1~5) (placeholder)."""
    raise NotImplementedError


def build_scorecard(
    instructor_id: str,
    lecture_date: str,
    item_scores: list[ItemScore],
    checklist: Checklist,
) -> InstructorScorecard:
    """단일 강의 → InstructorScorecard (placeholder)."""
    raise NotImplementedError


def compute_trend(points: list[TrendPoint]) -> tuple[float, str]:
    """날짜별 점수 → (선형회귀 기울기, 트렌드 라벨) (placeholder)."""
    raise NotImplementedError


def aggregate_weekly(scorecards: list[InstructorScorecard]) -> dict:
    """주차별 집계 (placeholder)."""
    raise NotImplementedError
