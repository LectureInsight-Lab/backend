"""스코어링 (5단계).

18개 항목 점수를 카테고리 평균 → 종합 점수로 집계하고,
다중 강의 분석 시 시계열 트렌드(선형 회귀)를 계산한다.

카테고리 가중치는 ``configs/checklist.yaml`` 의 ``category_weights`` 사용:
    structure 25% / concept 25% / practice 20% / language 15% / interaction 15%

트렌드 라벨:
    slope >= +0.05  → "improving"
    slope <= -0.05  → "declining"
    else            → "stable"
"""
from __future__ import annotations

from datetime import datetime

import numpy as np

from app.analysis.schemas import (
    CategoryScore,
    InstructorScorecard,
    ItemScore,
    TrendPoint,
)
from app.core.checklist import Checklist

TREND_THRESHOLD = 0.05


def category_scores(items: list[ItemScore], checklist: Checklist) -> list[CategoryScore]:
    """카테고리별 평균 점수 + 가중치 부착 (checklist 정의 순서 유지)."""
    result: list[CategoryScore] = []
    for category, weight in checklist.category_weights.items():
        members = [it for it in items if it.category == category]
        avg = sum(it.final_score for it in members) / len(members) if members else 0.0
        result.append(CategoryScore(category=category, score=round(avg, 4), weight=weight))
    return result


def overall_score(categories: list[CategoryScore]) -> float:
    """카테고리별 가중 평균 → 종합 점수 (1~5)."""
    total_weight = sum(c.weight for c in categories)
    if total_weight == 0:
        return 0.0
    weighted = sum(c.score * c.weight for c in categories)
    return round(weighted / total_weight, 4)


def build_scorecard(
    instructor_id: str,
    lecture_date: str,
    item_scores: list[ItemScore],
    checklist: Checklist,
) -> InstructorScorecard:
    """단일 강의 → InstructorScorecard."""
    cats = category_scores(item_scores, checklist)
    return InstructorScorecard(
        instructor_id=instructor_id,
        lecture_date=lecture_date,
        overall_score=overall_score(cats),
        category_scores=cats,
        item_scores=sorted(item_scores, key=lambda s: s.item_id),
    )


def compute_trend(points: list[TrendPoint]) -> tuple[float, str]:
    """날짜별 점수 → (선형회귀 기울기, 트렌드 라벨).

    x 축은 날짜 순서 인덱스(0..n-1) — 강의 회차당 점수 변화량을 기울기로 본다.
    점이 2개 미만이면 (0.0, "stable").
    """
    ordered = sorted(points, key=lambda p: p.date)
    if len(ordered) < 2:
        return 0.0, "stable"
    x = np.arange(len(ordered), dtype=float)
    y = np.array([p.score for p in ordered], dtype=float)
    slope = float(np.polyfit(x, y, 1)[0])
    if slope >= TREND_THRESHOLD:
        label = "improving"
    elif slope <= -TREND_THRESHOLD:
        label = "declining"
    else:
        label = "stable"
    return round(slope, 4), label


def attach_trend(scorecard: InstructorScorecard, history: list[InstructorScorecard]) -> InstructorScorecard:
    """동일 강사의 과거 스코어카드들로 트렌드를 계산해 부착(in-place)."""
    points = [TrendPoint(date=sc.lecture_date, score=sc.overall_score) for sc in history]
    slope, label = compute_trend(points)
    scorecard.trend_points = sorted(points, key=lambda p: p.date)
    scorecard.trend_slope = slope
    scorecard.trend_label = label
    return scorecard


def aggregate_weekly(scorecards: list[InstructorScorecard]) -> dict:
    """주차별 종합 점수 집계 → {"YYYY-Www": {"mean": float, "count": int, "dates": [...]}}"""
    buckets: dict[str, list[InstructorScorecard]] = {}
    for sc in scorecards:
        try:
            iso = datetime.strptime(sc.lecture_date, "%Y-%m-%d").isocalendar()
            key = f"{iso.year}-W{iso.week:02d}"
        except ValueError:
            key = sc.lecture_date
        buckets.setdefault(key, []).append(sc)

    out: dict[str, dict] = {}
    for key in sorted(buckets):
        group = buckets[key]
        mean = sum(sc.overall_score for sc in group) / len(group)
        out[key] = {
            "mean": round(mean, 4),
            "count": len(group),
            "dates": sorted(sc.lecture_date for sc in group),
        }
    return out
