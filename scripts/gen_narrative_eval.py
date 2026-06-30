"""종합 해설 정성평가용 텍스트 6종 생성 (김멋사).

저장된 스코어카드를 재사용해 '종합 해설(narrative)'만 생성한다 (라벨링 생략).
묶음/전체는 항목·카테고리 점수를 평균한 집계 스코어카드를 만든 뒤 종합 톤으로 생성.

생성 구성 (총 6개):
  1) 단일일 3개  : 02-02 / 02-03 / 02-04            (is_aggregate=False)
  2) 묶음 2개    : {02,03,04} / {25,26,27}            (is_aggregate=True, count=3)
  3) 전체 1개    : {02,03,04,25,26,27}                (is_aggregate=True, count=6)

각 파일에 요약(summary) + 종합 해설(overall_feedback) 을 기록한다.
스코어카드당 LLM 호출 약 1(explainer)+2(narrative) ≈ 3회.

Usage:
  python scripts/gen_narrative_eval.py
"""
from __future__ import annotations

import asyncio
from collections import defaultdict
from pathlib import Path

from app.analysis import explainer, narrative, scorer
from app.analysis.schemas import (
    CategoryScore,
    InstructorScorecard,
    ItemScore,
    TrendPoint,
)
from app.core import store

INSTRUCTOR = "김멋사"
OUT_DIR = Path("data/processed/narrative_eval")

# (출력 파일명 라벨, 멤버 날짜들, is_aggregate)
JOBS: list[tuple[str, list[str], bool]] = [
    ("단일_02-02", ["2026-02-02"], False),
    ("단일_02-03", ["2026-02-03"], False),
    ("단일_02-04", ["2026-02-04"], False),
    ("묶음_02-02~04", ["2026-02-02", "2026-02-03", "2026-02-04"], True),
    ("묶음_02-25~27", ["2026-02-25", "2026-02-26", "2026-02-27"], True),
    ("전체_6일", ["2026-02-02", "2026-02-03", "2026-02-04",
                  "2026-02-25", "2026-02-26", "2026-02-27"], True),
]


def _avg(xs: list[float]) -> float:
    return round(sum(xs) / len(xs), 2)


def _aggregate(cards: list[InstructorScorecard], label: str) -> InstructorScorecard:
    """여러 스코어카드를 항목/카테고리 평균으로 묶은 집계 카드 생성."""
    # 항목별 평균 (item_id 기준)
    by_item: dict[int, list[ItemScore]] = defaultdict(list)
    for c in cards:
        for it in c.item_scores:
            by_item[it.item_id].append(it)
    item_scores = [
        ItemScore(
            item_id=iid,
            name=group[0].name,
            category=group[0].category,
            item_type=group[0].item_type,
            final_score=_avg([g.final_score for g in group]),
            llm_score=_avg([g.llm_score for g in group]),
            bow_score=_avg([g.bow_score for g in group]),
            final_confidence=_avg([g.final_confidence for g in group]),
            evidence=group[0].evidence,           # explainer 가 재생성하므로 대표값
            reason=group[0].reason,
            strengths=group[0].strengths,
            improvements=group[0].improvements,
            needs_human_review=any(g.needs_human_review for g in group),
        )
        for iid, group in sorted(by_item.items())
    ]

    # 카테고리별 평균
    by_cat: dict[str, list[CategoryScore]] = defaultdict(list)
    for c in cards:
        for cs in c.category_scores:
            by_cat[cs.category].append(cs)
    category_scores = [
        CategoryScore(
            category=cat,
            score=_avg([g.score for g in group]),
            weight=group[0].weight,
        )
        for cat, group in by_cat.items()
    ]

    card = InstructorScorecard(
        instructor_id=INSTRUCTOR,
        lecture_date=label,
        overall_score=_avg([c.overall_score for c in cards]),
        category_scores=category_scores,
        item_scores=item_scores,
    )
    # 추이(개선/유지/하락) 부착 — 종합 해설 톤 강화
    scorer.attach_trend(
        card,
        sorted(cards, key=lambda c: c.lecture_date),
    )
    return card


async def _one_job(label: str, dates: list[str], is_aggregate: bool) -> str:
    cards = [store.get(store.scorecard_id(INSTRUCTOR, d)) for d in dates]
    missing = [d for d, c in zip(dates, cards) if c is None]
    if missing:
        raise SystemExit(f"[gen] 스코어카드 없음: {INSTRUCTOR} {missing}")

    if is_aggregate:
        date_label = f"{dates[0]} ~ {dates[-1]} ({len(dates)}일)"
        card = _aggregate(cards, date_label)
        count = len(dates)
    else:
        date_label = dates[0]
        card = cards[0]
        count = 1

    # 1) 항목 해설 재생성 (in-place) → narrative 가 it.reason 을 활용
    await explainer.attach_explanations(card)
    # 2) 종합 해설 + 요약
    nar = await narrative.generate_narrative(card, is_aggregate=is_aggregate, lecture_count=count)

    mode = "종합(여러 강의 평균)" if is_aggregate else "단일 강의"
    body = (
        f"# 강사: {INSTRUCTOR}\n"
        f"# 평가 구성: {label}\n"
        f"# 대상 일자: {date_label}\n"
        f"# 모드: {mode} | 강의 수: {count} | 종합 점수: {card.overall_score}/5.0\n"
        f"{'=' * 60}\n\n"
        f"[요약]\n{nar['summary']}\n\n"
        f"[종합 해설]\n{nar['overall_feedback']}\n"
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{INSTRUCTOR}_{label}.txt"
    out.write_text(body, encoding="utf-8")
    print(f"[gen] ✓ {out}  (overall={card.overall_score}, count={count})")
    return str(out)


async def _main() -> None:
    print(f"[gen] 종합 해설 평가용 {len(JOBS)}개 생성 시작 — {INSTRUCTOR}")
    for label, dates, is_agg in JOBS:
        await _one_job(label, dates, is_agg)
    print(f"\n[gen] 완료 — 출력 폴더: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    asyncio.run(_main())
