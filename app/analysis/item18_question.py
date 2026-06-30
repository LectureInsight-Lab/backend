"""
app/analysis/item18_question.py — 항목 18 질문 응답 충분성 (LLM 채점)

담당자: 박다현

흐름: preprocessing.item18_question.run(df) 로 Q-A 페어 chunk 획득
      → 루브릭 프롬프트로 Gemini 1회 호출 → LLM 이 1~5점 직접 산출.
채점: 질문 응답의 충분성. 루브릭은 동일명 YAML 참조.

입력: 단일 강의 kss DataFrame
출력: {"evidence": list|None, "reason": str, "final_score": int}
"""
from __future__ import annotations

import asyncio

import pandas as pd

from app.analysis import rubric_llm
from app.preprocessing.item18_question import run as build_chunk

MODULE_NAME = "question"
ITEM_ID = 18
_STEM = "item18_question"


def _score(parsed: dict) -> tuple[int, str]:
    try:
        score = int(round(float(parsed.get("score", 1))))
    except (TypeError, ValueError):
        score = 1
    score = max(1, min(5, score))
    reason = (
        f"qa_count={parsed.get('qa_count')}, "
        f"fully_answered={parsed.get('fully_answered')}"
    )
    return score, reason


async def run(df: pd.DataFrame, concurrency: int = 10, mode: str = "") -> dict:
    """단일 강의 kss DataFrame → 질문 응답 충분성 점수."""
    chunk = build_chunk(df).get("chunk", "")
    if not chunk:
        return rubric_llm.build(1, "Q-A 페어 없음", None, mode)
    prompt = rubric_llm.load_prompt(_STEM)
    sem = asyncio.Semaphore(concurrency)
    parsed = await rubric_llm.judge(prompt["system"], rubric_llm.render_user(prompt, chunk), sem)
    score, reason = _score(parsed)
    return rubric_llm.build(score, reason, [parsed], mode)


# ─── CLI ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json

    from app.preprocessing.utils import parse_and_split

    parser = argparse.ArgumentParser(description="항목 18: 질문 응답 충분성 LLM 채점")
    parser.add_argument("txt_path", help="원본 강의 텍스트 파일 경로 (단일 강의)")
    parser.add_argument("--mode", default="", help='"debug" 시 evidence 포함')
    args = parser.parse_args()

    _df = parse_and_split(args.txt_path)
    result = asyncio.run(run(_df, mode=args.mode))
    print(json.dumps(result, ensure_ascii=False, indent=2))
