"""
app/analysis/item04_learning_objectives.py — 항목 4 학습 목표 안내 (LLM 채점)

담당자: 심소민

흐름: preprocessing.item04_learning_objectives.run(df)로 도입부 chunk 획득
      → 루브릭 프롬프트로 Gemini 1회 호출 → LLM이 1~5점 직접 산출.
채점: 학습 목표 구체성 × 순서 제시 (3·4점은 순서 제시 여부로 구분). 루브릭은 동일명 YAML 참조.

입력: 단일 강의 kss DataFrame
출력: {"evidence": list|None, "reason": str, "final_score": int}
"""
from __future__ import annotations

import asyncio

import pandas as pd

from app.analysis import rubric_llm
from app.preprocessing.item04_learning_objectives import run as build_chunks

MODULE_NAME = "learning_objectives"
ITEM_ID = 4
_STEM = "item04_learning_objectives"


def _score(parsed: dict) -> tuple[int, str]:
    """LLM 응답(점수 직접 산출) → (final_score, reason)."""
    try:
        score = int(round(float(parsed.get("score", 1))))
    except (TypeError, ValueError):
        score = 1
    score = max(1, min(5, score))
    reason = (
        f"goal_stated={parsed.get('goal_stated')}, "
        f"order_mentioned={parsed.get('order_mentioned')}"
    )
    return score, reason


async def run(df: pd.DataFrame, concurrency: int = 10, mode: str = "") -> dict:
    """단일 강의 kss DataFrame → 학습 목표 안내 점수."""
    chunk = build_chunks(df).get("chunk", "")
    prompt = rubric_llm.load_prompt(_STEM)
    sem = asyncio.Semaphore(concurrency)
    parsed = await rubric_llm.judge(prompt["system"], rubric_llm.render_user(prompt, chunk), sem)
    score, reason = _score(parsed)
    return rubric_llm.build(score, reason, [parsed], mode)


# ─── CLI (standalone, 실제 LLM 호출) ──────────────────────────────

if __name__ == "__main__":
    import argparse
    import json

    from app.preprocessing.utils import parse_and_split

    parser = argparse.ArgumentParser(description="항목 4: 학습 목표 안내 LLM 채점")
    parser.add_argument("txt_path", help="원본 강의 텍스트 파일 경로 (단일 강의)")
    parser.add_argument("--mode", default="", help='"debug" 시 evidence 포함')
    args = parser.parse_args()

    _df = parse_and_split(args.txt_path)
    result = asyncio.run(run(_df, mode=args.mode))
    print(json.dumps(result, ensure_ascii=False, indent=2))
