"""
app/analysis/item10_example_coverage.py — 항목 10 비유/예시 활용 (LLM 채점)

담당자: 심소민

흐름: preprocessing.item10_example_coverage.run(df)로 '개념' 청크 목록 획득
      → 청크별 Gemini 호출(예시 필요 여부 + 0~5점, 불필요 시 "NA")
      → N/A 제외 점수들의 평균 → 1~5점.
채점: 평균 점수를 반올림(half-up) 후 1~5로 클램프. 평가 대상 예시가 없으면 "N/A".
  ⚠️ 평균의 반올림/내림 규칙은 gold set 작업 중 확정 예정(현재 half-up).

입력: 단일 강의 labeled DataFrame
출력: {"evidence": list|None, "reason": str, "final_score": int|"N/A"}
"""
from __future__ import annotations

import asyncio
import math

import pandas as pd

from app.analysis import rubric_llm
from app.preprocessing.item10_example_coverage import run as build_chunks

MODULE_NAME = "example_coverage"
ITEM_ID = 10
_STEM = "item10_example_coverage"


def _score(parsed_list: list[dict]) -> tuple[object, str, list[dict]]:
    """청크별 LLM 응답 → (final_score|"N/A", reason, evidence_items).

    score=="NA"(예시 불필요)는 평가 대상에서 제외. 나머지 0~5점의 평균을 half-up 반올림.
    """
    applicable: list[int] = []
    for r in parsed_list:
        s = r.get("score")
        if isinstance(s, str) and s.strip().upper() in {"NA", "N/A"}:
            continue
        try:
            applicable.append(int(round(float(s))))
        except (TypeError, ValueError):
            continue

    if not applicable:
        return "N/A", "예시가 필요한 개념 없음", []

    mean = sum(applicable) / len(applicable)
    final = max(1, min(5, math.floor(mean + 0.5)))  # half-up → 1~5 클램프
    reason = f"mean={mean:.2f} (n={len(applicable)}, NA 제외)"
    return final, reason, parsed_list


async def run(df: pd.DataFrame, concurrency: int = 10, mode: str = "") -> dict:
    """단일 강의 labeled DataFrame → 비유/예시 활용 점수."""
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

    parser = argparse.ArgumentParser(description="항목 10: 비유/예시 활용 LLM 채점")
    parser.add_argument("txt_path", help="원본 강의 텍스트 파일 경로 (단일 강의)")
    parser.add_argument("--concurrency", "-c", type=int, default=10)
    parser.add_argument("--mode", default="", help='"debug" 시 evidence 포함')
    args = parser.parse_args()

    _txt = Path(args.txt_path)
    parse_and_split(_txt)
    _labeled = pd.DataFrame(label_from_csv(PROCESSED_DIR / f"{_txt.stem}_kss.csv", concurrency=args.concurrency))
    result = asyncio.run(run(_labeled, concurrency=args.concurrency, mode=args.mode))
    print(json.dumps(result, ensure_ascii=False, indent=2))
