"""tests/test_scorer.py — 카테고리 가중 평균 + 시계열 트렌드."""
from __future__ import annotations

from app.analysis import scorer
from app.analysis.schemas import ItemScore, TrendPoint
from app.core.checklist import load_checklist

CL = load_checklist()


def _items(score_by_category: dict[str, float]) -> list[ItemScore]:
    out: list[ItemScore] = []
    for it in CL.items:
        s = score_by_category.get(it.category, 3.0)
        out.append(ItemScore(item_id=it.id, name=it.name, category=it.category,
                             item_type=it.item_type.value, final_score=s, llm_score=s,
                             bow_score=0.0, final_confidence=0.7, evidence="", strengths="",
                             improvements=""))
    return out


def test_category_scores_average():
    items = _items({"structure": 4.0, "concept": 2.0})
    cats = {c.category: c for c in scorer.category_scores(items, CL)}
    assert cats["structure"].score == 4.0
    assert cats["concept"].score == 2.0
    assert cats["structure"].weight == CL.category_weights["structure"]


def test_overall_is_weighted_average():
    items = _items({})                      # 전부 3.0
    cats = scorer.category_scores(items, CL)
    assert abs(scorer.overall_score(cats) - 3.0) < 1e-6


def test_build_scorecard_shapes():
    card = scorer.build_scorecard("inst1", "2026-02-02", _items({}), CL)
    assert len(card.item_scores) == 18
    assert len(card.category_scores) == 5
    assert 1.0 <= card.overall_score <= 5.0


def test_trend_labels():
    up = [TrendPoint(date=f"2026-02-0{i}", score=v) for i, v in enumerate([2.0, 3.0, 4.0], 1)]
    down = [TrendPoint(date=f"2026-02-0{i}", score=v) for i, v in enumerate([4.0, 3.0, 2.0], 1)]
    flat = [TrendPoint(date=f"2026-02-0{i}", score=3.0) for i in range(1, 4)]
    assert scorer.compute_trend(up)[1] == "improving"
    assert scorer.compute_trend(down)[1] == "declining"
    assert scorer.compute_trend(flat)[1] == "stable"
    assert scorer.compute_trend([TrendPoint(date="2026-02-01", score=3.0)]) == (0.0, "stable")


def test_weekly_aggregation():
    cards = [scorer.build_scorecard("i", d, _items({}), CL) for d in ("2026-02-02", "2026-02-03")]
    weekly = scorer.aggregate_weekly(cards)
    assert sum(v["count"] for v in weekly.values()) == 2
