"""
app/analysis/item13_example_relevance.py — 항목 13 예시 적절성 (LLM 채점)

담당자: 심소민

흐름: preprocessing.item13_example_relevance.run(df)로 예시 후보 청크 목록 획득
      → 청크별 Gemini 호출(실제 예시 여부 + level + relevance)
      → level_ok_ratio / relevance_high_ratio 집계 → 2D joint threshold로 1~5점.
채점:
  level_ok_ratio      = "적절" 수 / 전체 예시 수 × 100
  relevance_high_ratio= "직접연관" 수 / 전체 예시 수 × 100

입력: 단일 강의 labeled DataFrame
출력: {"evidence": list|None, "reason": str, "final_score": int}
"""
from __future__ import annotations

import asyncio

import pandas as pd

from app.analysis import rubric_llm
from app.preprocessing.item13_example_relevance import run as build_chunks

MODULE_NAME = "example_relevance"
ITEM_ID = 13
_STEM = "item13_example_relevance"


def _joint_score(level_ok: float, rel_high: float, n: int) -> int:
    """level_ok_ratio × relevance_high_ratio 2D joint threshold → 1~5."""
    if n == 0:
        return 1  # 예시 없음
    if level_ok >= 80 and rel_high >= 70:
        return 5
    if level_ok >= 70 and rel_high >= 50:
        return 4
    if level_ok >= 50 and rel_high >= 30:
        return 3
    if level_ok < 50:
        return 2
    # gap: level_ok>=50 이지만 relevance 부족 — 연관성 미달로 2점 처리 (TODO: gold set으로 확정)
    return 2


def _score(parsed_list: list[dict]) -> tuple[int, str, list[dict]]:
    """청크별 LLM 응답 → (final_score, reason, evidence_items). is_example=true만 집계."""
    examples = [r for r in parsed_list if r.get("is_example")]
    n = len(examples)
    if n == 0:
        return 1, "확인된 예시 없음", examples

    level_ok = sum(1 for r in examples if r.get("level") == "적절") / n * 100
    rel_high = sum(1 for r in examples if r.get("relevance") == "직접연관") / n * 100
    final = _joint_score(level_ok, rel_high, n)
    reason = f"level_ok={level_ok:.0f}% relevance_high={rel_high:.0f}% (n={n})"
    return final, reason, examples


async def run(df: pd.DataFrame, concurrency: int = 10, mode: str = "") -> dict:
    """단일 강의 labeled DataFrame → 예시 적절성 점수."""
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
    from pathlib import Path

    from app.preprocessing.utils import PROCESSED_DIR, label_from_csv, parse_and_split

    parser = argparse.ArgumentParser(description="항목 13: 예시 적절성 LLM 채점")
    parser.add_argument("txt_path", help="원본 강의 텍스트 파일 경로 (단일 강의)")
    parser.add_argument("--concurrency", "-c", type=int, default=10)
    parser.add_argument("--mode", default="", help='"debug" 시 evidence 포함')
    args = parser.parse_args()

    _txt = Path(args.txt_path)
    parse_and_split(_txt)
    _labeled = pd.DataFrame(label_from_csv(PROCESSED_DIR / f"{_txt.stem}_kss.csv", concurrency=args.concurrency))
    result = asyncio.run(run(_labeled, concurrency=args.concurrency, mode=args.mode))
    print(json.dumps(result, ensure_ascii=False, indent=2))
