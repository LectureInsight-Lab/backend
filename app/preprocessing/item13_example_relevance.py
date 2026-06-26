"""
app/preprocessing/item13_example_relevance.py — 항목 13 예시 적절성

담당자: 심소민

라벨링된 청크(labeled)에서 예시 후보를 추출한다.
후보 = llm_label == '예시' 청크 ∪ 예시 표현 regex가 매칭된 청크 (text 기준 dedup).
실제 예시 여부 판별과 수강생 수준 대비 적절성(level·relevance) 채점은
app/analysis(LLM)가 담당한다.

흐름: run(df) → '예시' 청크 + regex 매칭 청크 합집합 → dedup → chunk 목록
입력: 단일 강의 labeled DataFrame (label_from_csv 결과). 필수 컬럼: llm_label, text
출력: {"chunks": [{"request_id": str, "text": str, "context": {...}}, ...]}
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

MODULE_NAME = "example_relevance"
ITEM_ID = 13

# 예시 후보 탐지용 regex (라벨 청크 텍스트에 적용)
_EXAMPLE_REGEX = re.compile(
    r"예를\s*들[어면]|예시[로를]?|예를\s*들면|비유[하자면로]?|"
    r"예컨대|실제[로]?\s*예|예를\s*들어서|쉽게\s*말[하해]면",
    re.IGNORECASE,
)


# ─── 전처리 → chunk 생성 ──────────────────────────────────────────

def _example_candidates(labeled: pd.DataFrame) -> list[dict]:
    """'예시' 라벨 청크 ∪ 예시 regex 매칭 청크를 후보로 추출한다 (text dedup)."""
    candidates: list[dict] = []
    seen_texts: set[str] = set()

    for i, row in labeled.iterrows():
        text = str(row.get("text", "")).strip()
        if not text or text in seen_texts:
            continue

        label = row.get("llm_label")
        is_anchored = label == "예시"
        is_regex = bool(_EXAMPLE_REGEX.search(text))
        if not (is_anchored or is_regex):
            continue

        seen_texts.add(text)
        candidates.append({
            "request_id": f"example_{i}",
            "text": text,
            "context": {
                "source": "anchored" if is_anchored else "regex",
                "llm_label": str(label),
            },
        })

    return candidates


def run(df: pd.DataFrame) -> dict:
    """단일 강의 labeled DataFrame → 예시 적절성 평가용 chunk 목록.

    Args:
        df: 단일 강의 라벨 청크 DataFrame. 필수 컬럼: llm_label, text.

    Returns:
        {"chunks": [{"request_id", "text", "context"}, ...]}
    """
    return {"chunks": _example_candidates(df)}


# ─── CLI 진입점 (standalone 테스트, LLM 호출 없음) ────────────────

if __name__ == "__main__":
    import argparse

    from app.preprocessing.utils import PROCESSED_DIR, label_from_csv, parse_and_split

    parser = argparse.ArgumentParser(description="항목 13: 예시 적절성 chunk 추출")
    parser.add_argument("txt_path", help="원본 강의 텍스트 파일 경로 (단일 강의)")
    parser.add_argument("--output", "-o", default=None, help="결과 JSON 저장 경로 (생략 시 stdout)")
    parser.add_argument("--concurrency", "-c", type=int, default=10, help="라벨링 동시 요청 수 (기본 10)")
    args = parser.parse_args()

    _txt = Path(args.txt_path)
    parse_and_split(_txt)
    _csv = PROCESSED_DIR / f"{_txt.stem}_kss.csv"
    _labeled = pd.DataFrame(label_from_csv(_csv, concurrency=args.concurrency))
    result = run(_labeled)

    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"저장 완료: {args.output}")
    else:
        print(output)
