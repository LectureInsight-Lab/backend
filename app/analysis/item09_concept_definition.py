"""
app/analysis/item09_concept_definition.py — 항목 9 개념 정의 (LLM 채점)

담당자: 심소민

흐름: preprocessing.item09_concept_definition.run(df)로 '개념' 청크 목록 획득
      → 청크별 Gemini 호출(개념명 정규화 + 정의 명확성 1~5)
      → 동일 개념명 dedup(첫 등장) → definition_rate 집계 → 1~5점.
채점: definition_rate = (정의수준 4점 이상 개념 수 / 전체 고유 개념 수) × 100.

입력: 단일 강의 labeled DataFrame
출력: {"evidence": list|None, "reason": str, "final_score": int|"N/A"}
"""
from __future__ import annotations

import asyncio

import pandas as pd

from app.analysis import rubric_llm
from app.preprocessing.utils import apply_score_rubric
from app.preprocessing.item09_concept_definition import run as build_chunks

MODULE_NAME = "concept_definition"
ITEM_ID = 9
_STEM = "item09_concept_definition"

# definition_rate(%) → final_score(1-5)
_RATE_THRESHOLDS: list[tuple[float, int]] = [(80, 5), (60, 4), (40, 3), (20, 2), (0, 1)]


def _score(parsed_list: list[dict]) -> tuple[object, str, list[dict]]:
    """청크별 LLM 응답 → (final_score|"N/A", reason, evidence_items).

    동일 concept_name 첫 등장만 남겨 중복 제거 후 definition_rate 집계.
    """
    seen: set[str] = set()
    deduped: list[dict] = []
    for r in parsed_list:
        name = str(r.get("concept_name", "")).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        deduped.append(r)

    if not deduped:
        return "N/A", "평가 가능한 개념 없음", []

    total = len(deduped)
    high = sum(1 for r in deduped if _as_int(r.get("definition_score")) >= 4)
    rate = high / total * 100
    final = apply_score_rubric(rate, _RATE_THRESHOLDS)
    reason = f"definition_rate={rate:.1f}% ({high}/{total}개 개념이 4점 이상)"
    return final, reason, deduped


def _as_int(v) -> int:
    try:
        return int(round(float(v)))
    except (TypeError, ValueError):
        return 0


async def run(df: pd.DataFrame, concurrency: int = 10, mode: str = "") -> dict:
    """단일 강의 labeled DataFrame → 개념 정의 점수."""
    chunks = build_chunks(df).get("chunks", [])
    prompt = rubric_llm.load_prompt(_STEM)
    sem = asyncio.Semaphore(concurrency)
    parsed_list = await asyncio.gather(
        *[
            rubric_llm.judge(prompt["system"], rubric_llm.render_user(prompt, c["text"]), sem)
            for c in chunks
        ]
    )
    final, reason, evidence = _score(list(parsed_list))
    return rubric_llm.build(final, reason, evidence, mode)


# ─── CLI (standalone, 실제 LLM 호출) ──────────────────────────────

if __name__ == "__main__":
    import argparse
    import json

    from app.preprocessing.utils import PROCESSED_DIR, label_from_csv, parse_and_split
    from pathlib import Path

    parser = argparse.ArgumentParser(description="항목 9: 개념 정의 LLM 채점")
    parser.add_argument("txt_path", help="원본 강의 텍스트 파일 경로 (단일 강의)")
    parser.add_argument("--concurrency", "-c", type=int, default=10)
    parser.add_argument("--mode", default="", help='"debug" 시 evidence 포함')
    args = parser.parse_args()

    _txt = Path(args.txt_path)
    parse_and_split(_txt)
    _labeled = pd.DataFrame(label_from_csv(PROCESSED_DIR / f"{_txt.stem}_kss.csv", concurrency=args.concurrency))
    result = asyncio.run(run(_labeled, concurrency=args.concurrency, mode=args.mode))
    print(json.dumps(result, ensure_ascii=False, indent=2))
