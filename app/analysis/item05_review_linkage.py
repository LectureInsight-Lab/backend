"""
app/analysis/item05_review_linkage.py — 항목 5 전날 복습 연계 (LLM 채점)

담당자: 심소민

흐름: preprocessing.item05_review_linkage.run(df)로 도입부 복습 컨텍스트 chunk 획득
      → 루브릭 프롬프트로 Gemini 1회 호출 → LLM이 1~5점 직접 산출.
채점: 이전 강의 복습(구체/추상) × 오늘 주제와의 논리적 연결성(관계유형). 루브릭은 동일명 YAML.

입력: 단일 강의 kss DataFrame
출력: {"evidence": list|None, "reason": str, "final_score": int}
"""
from __future__ import annotations

import asyncio

import pandas as pd

from app.analysis import rubric_llm
from app.preprocessing.item05_review_linkage import run as build_chunks

MODULE_NAME = "review_linkage"
ITEM_ID = 5
_STEM = "item05_review_linkage"


def _score(parsed: dict) -> tuple[int, str]:
    """LLM 응답(점수 직접 산출) → (final_score, reason)."""
    try:
        score = int(round(float(parsed.get("score", 1))))
    except (TypeError, ValueError):
        score = 1
    score = max(1, min(5, score))
    reason = (
        f"hit_detected={parsed.get('hit_detected')}, "
        f"relation_type={parsed.get('relation_type')}"
    )
    return score, reason


async def run(df: pd.DataFrame, concurrency: int = 10, mode: str = "") -> dict:
    """단일 강의 kss DataFrame → 전날 복습 연계 점수."""
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

    parser = argparse.ArgumentParser(description="항목 5: 전날 복습 연계 LLM 채점")
    parser.add_argument("txt_path", help="원본 강의 텍스트 파일 경로 (단일 강의)")
    parser.add_argument("--mode", default="", help='"debug" 시 evidence 포함')
    args = parser.parse_args()

    _df = parse_and_split(args.txt_path)
    result = asyncio.run(run(_df, mode=args.mode))
    print(json.dumps(result, ensure_ascii=False, indent=2))
