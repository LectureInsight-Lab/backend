"""저장된 스코어카드로 '해설'만 재생성 (라벨링 파이프라인 생략).

비싼 청크 라벨링(Gemini ×수십)은 이미 끝나 스코어카드 JSON에 저장돼 있다.
이 스크립트는 그 스코어카드를 로드해 표현 레이어(해설)만 다시 만든다:
  - explainer.attach_explanations : 항목별 strengths/improvements/근거 (Gemini 1회)
  - narrative.generate_narrative  : 종합 해설 + 요약          (Gemini 1~2회)
→ 스코어카드 1건당 LLM 호출 2~3회. 점수(structured)는 손대지 않는다.

Usage:
  # 단일 강의
  python scripts/regen_explanations.py 이수민 2026-02-02

  # 강사 전체(여러 날짜) 종합 해설
  python scripts/regen_explanations.py 이수민 --aggregate

  # 재생성 결과를 스코어카드 JSON 에 덮어쓰기
  python scripts/regen_explanations.py 이수민 2026-02-02 --save
"""
from __future__ import annotations

import argparse
import asyncio
import json

from app.analysis import explainer, narrative
from app.core import store


async def _run(instructor: str, date: str | None, aggregate: bool, save: bool) -> None:
    if aggregate or date is None:
        cards = store.list_by_instructor(instructor)
        if not cards:
            raise SystemExit(f"[regen] 스코어카드 없음 — instructor={instructor}")
        # 종합은 가장 최근 날짜 카드를 대표로 해설 생성(점수는 그대로)
        card = sorted(cards, key=lambda c: c.lecture_date)[-1]
        is_aggregate, lecture_count = True, len(cards)
        print(f"[regen] 종합 모드 — {instructor}, {lecture_count}개 강의, 대표일={card.lecture_date}")
    else:
        card = store.get(store.scorecard_id(instructor, date))
        if card is None:
            raise SystemExit(f"[regen] 스코어카드 없음 — {instructor}/{date}")
        is_aggregate, lecture_count = False, 1
        print(f"[regen] 단일 모드 — {instructor}/{date}, overall={card.overall_score}")

    # 1) 항목별 해설 (in-place)
    await explainer.attach_explanations(card)

    # 2) 종합 해설 + 요약
    nar = await narrative.generate_narrative(card, is_aggregate=is_aggregate, lecture_count=lecture_count)

    print("\n===== 항목별 해설 =====")
    for it in card.item_scores:
        print(f"[{it.item_id:>2}] {it.name} ({it.final_score})")
        print(f"     근거: {it.evidence}")
        print(f"     코멘트: {it.reason}")
        print(f"     강점: {it.strengths}")
        print(f"     개선: {it.improvements}\n")

    print("===== 종합 해설 =====")
    print(json.dumps(nar, ensure_ascii=False, indent=2))

    if save:
        cid = store.save(card)
        print(f"\n[regen] 저장 완료 → {cid}")


def main() -> None:
    p = argparse.ArgumentParser(description="저장된 스코어카드로 해설 재생성")
    p.add_argument("instructor", help="강사 ID (예: 이수민)")
    p.add_argument("date", nargs="?", default=None, help="강의 일자 YYYY-MM-DD (생략 시 종합)")
    p.add_argument("--aggregate", action="store_true", help="강사 전체 날짜 종합 해설")
    p.add_argument("--save", action="store_true", help="재생성 해설을 스코어카드 JSON 에 덮어쓰기")
    args = p.parse_args()
    asyncio.run(_run(args.instructor, args.date, args.aggregate, args.save))


if __name__ == "__main__":
    main()
