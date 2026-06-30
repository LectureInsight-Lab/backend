"""
app/preprocessing/item04_learning_objectives.py — 항목 4 학습 목표 안내

담당자: 심소민

도입부 7분 발화를 하나의 chunk(LLM 입력 payload)로 묶어 반환한다.
LLM 호출·채점은 app/analysis가 담당한다.

흐름: run(df) → 도입부 7분 추출 → chunk 문자열
입력: 단일 강의 kss DataFrame (parse_and_split 결과). 필수 컬럼: elapsed_sec, text_raw
출력: {"chunk": str}  — 도입부 전체 발화 텍스트
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app.preprocessing.utils import (
    ensure_sentence_id,
    extract_intro_segment,
    format_sentences_as_text,
)

MODULE_NAME = "learning_objectives"
ITEM_ID = 4
INTRO_WINDOW_SEC = 420  # 도입부 7분 (goldset opening recall 평가 기반; 30분→7분 축소, recall 1.0 유지)


# ─── 전처리 → chunk 생성 ──────────────────────────────────────────

def _intro_transcript(df: pd.DataFrame, window_sec: int = INTRO_WINDOW_SEC) -> str:
    """도입부 window_sec 구간 발화를 '[elapsed_sec]s text' 형태 텍스트로 결합한다."""
    intro = extract_intro_segment(ensure_sentence_id(df.copy()), window_sec=window_sec)
    sentences = [
        {
            "sentence_id": int(r["sentence_id"]),
            "elapsed_sec": float(r["elapsed_sec"]),
            "text": str(r["text_raw"]),
        }
        for _, r in intro.iterrows()
    ]
    return format_sentences_as_text(sentences)


def run(df: pd.DataFrame) -> dict:
    """단일 강의 kss DataFrame → 학습 목표 평가용 chunk.

    Args:
        df: 단일 강의 발화 DataFrame. 필수 컬럼: elapsed_sec, text_raw.

    Returns:
        {"chunk": str} — 도입부 7분 전체 발화 텍스트.
    """
    transcript = _intro_transcript(df)
    return {"chunk": f"[강의 도입부 발화]\n{transcript}"}


# ─── CLI 진입점 (standalone 테스트, LLM 호출 없음) ────────────────

if __name__ == "__main__":
    import argparse

    from app.preprocessing.utils import parse_and_split

    parser = argparse.ArgumentParser(description="항목 4: 학습 목표 안내 chunk 추출")
    parser.add_argument("txt_path", help="원본 강의 텍스트 파일 경로 (단일 강의)")
    parser.add_argument("--output", "-o", default=None, help="결과 JSON 저장 경로 (생략 시 stdout)")
    args = parser.parse_args()

    _df = parse_and_split(args.txt_path)
    result = run(_df)

    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"저장 완료: {args.output}")
    else:
        print(output)
