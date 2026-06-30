"""
app/preprocessing/item05_review_linkage.py — 항목 5 전날 복습 연계

담당자: 심소민

도입부 30분에서 복습 키워드(REVIEW_KEYWORDS)를 탐지해 첫 히트 기준 [0, +6분] 시간 윈도우를
chunk로 반환한다. 키워드 미탐지 시 anchor=강의 시작(t=0)으로 본 [0, +6분] 윈도우를 fallback 으로 쓴다(동일 규칙).
LLM 호출·채점은 app/analysis가 담당한다. (윈도우 길이는 goldset opening recall 평가 기반)

흐름: run(df) → 도입부 30분서 첫 키워드 탐지 → 첫 히트 [0, +6분] 시간 윈도우(또는 fallback)
입력: 단일 강의 kss DataFrame. 필수 컬럼: elapsed_sec, text_raw
출력: {"chunk": str, "context": {"hit_detected": bool, "keyword": str|None, ...}}
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app.preprocessing.utils import (
    detect_keywords,
    ensure_sentence_id,
    extract_intro_segment,
    extract_time_window,
    format_sentences_as_text,
)

MODULE_NAME = "review_linkage"
ITEM_ID = 5
INTRO_WINDOW_SEC = 1800   # 첫 복습 키워드 탐색 윈도우 (도입부 30분)
CONTEXT_BEFORE_SEC = 0    # 첫 히트 이전 컨텍스트 (opening recall 에 무영향 → 0)
CONTEXT_AFTER_SEC = 360   # 첫 히트 이후 6분 — 도입부 복습이 길게 이어지는 날 커버 (goldset 평가 knee)

# 복습 연계 탐지 키워드 — 도메인별 재검증 필요 (일반 과목 강의 적용 시 미탐지 위험)
REVIEW_KEYWORDS = ["지난", "복습", "어제", "저번", "이전", "배웠", "했던", "다뤘"]


# ─── 전처리 → chunk 생성 ──────────────────────────────────────────

def run(df: pd.DataFrame) -> dict:
    """단일 강의 kss DataFrame → 복습 연계 평가용 chunk.

    Args:
        df: 단일 강의 발화 DataFrame. 필수 컬럼: elapsed_sec, text_raw.

    Returns:
        {"chunk": str, "context": {hit_detected, keyword, ...}}
    """
    sentences = ensure_sentence_id(df.copy())
    intro = extract_intro_segment(sentences, window_sec=INTRO_WINDOW_SEC)
    hits = detect_keywords(intro, REVIEW_KEYWORDS)

    if hits:
        first_hit = hits[0]
        # 첫 히트 기준 시간 윈도우 — 전체 sentences 에서 끊어 +6분이 도입부 30분 경계에 잘리지 않게
        context_sentences = extract_time_window(
            sentences, first_hit["elapsed_sec"], CONTEXT_BEFORE_SEC, CONTEXT_AFTER_SEC
        )
        context_text = format_sentences_as_text(context_sentences)
        chunk = (
            f"[탐지된 키워드] '{first_hit['keyword']}' @ {first_hit['elapsed_sec']:.0f}s\n\n"
            f"[발화 컨텍스트]\n{context_text}"
        )
        context = {
            "hit_detected": True,
            "keyword": first_hit["keyword"],
            "hit_sentence_id": first_hit["sentence_id"],
            "hit_elapsed_sec": first_hit["elapsed_sec"],
            "total_hits": len(hits),
        }
    else:
        # 키워드 미탐지 → anchor=강의 시작(t=0)으로 본 [0, +6분] 윈도우 (메인 경로와 동일 규칙)
        fallback_sentences = extract_time_window(sentences, 0.0, 0, CONTEXT_AFTER_SEC)
        fallback_text = format_sentences_as_text(fallback_sentences)
        chunk = f"[도입부 0~{CONTEXT_AFTER_SEC // 60}분 발화]\n{fallback_text}"
        context = {
            "hit_detected": False,
            "keyword": None,
            "hit_sentence_id": None,
            "hit_elapsed_sec": None,
            "total_hits": 0,
        }

    return {"chunk": chunk, "context": context}


# ─── CLI 진입점 (standalone 테스트, LLM 호출 없음) ────────────────

if __name__ == "__main__":
    import argparse

    from app.preprocessing.utils import parse_and_split

    parser = argparse.ArgumentParser(description="항목 5: 전날 복습 연계 chunk 추출")
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
