"""
app/preprocessing/item10_example_coverage.py — 항목 10 비유/예시 활용(제공률)

담당자: 심소민

라벨링된 청크(labeled) 중 llm_label == '개념' 청크 텍스트를 그대로 컨텍스트로 사용해
"이 개념에 예시·비유·실습이 제공되었는가"를 평가할 chunk 목록을 반환한다.
예시 제공 여부·품질 판단과 coverage 채점은 app/analysis(LLM)가 담당한다.

흐름: run(df) → '개념' 청크 추출 → 청크 텍스트를 컨텍스트로 chunk 목록 생성
입력: 단일 강의 labeled DataFrame (label_from_csv 결과). 필수 컬럼: llm_label, text
출력: {"chunks": [{"request_id": str, "text": str, "context": {...}}, ...]}
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

MODULE_NAME = "example_coverage"
ITEM_ID = 10


# ─── 전처리 → chunk 생성 ──────────────────────────────────────────

def _concept_chunks(labeled: pd.DataFrame) -> list[dict]:
    """llm_label == '개념' 청크 텍스트를 예시 제공 평가용 chunk로 변환한다."""
    concept_chunks = labeled[labeled["llm_label"] == "개념"].reset_index(drop=True)

    result: list[dict] = []
    for i, row in concept_chunks.iterrows():
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        result.append({
            "request_id": f"concept_{i}",
            "text": text,
            "context": {"source": str(row.get("file", ""))},
        })

    return result


def run(df: pd.DataFrame) -> dict:
    """단일 강의 labeled DataFrame → 예시 제공(coverage) 평가용 chunk 목록.

    Args:
        df: 단일 강의 라벨 청크 DataFrame. 필수 컬럼: llm_label, text.

    Returns:
        {"chunks": [{"request_id", "text", "context"}, ...]}
    """
    return {"chunks": _concept_chunks(df)}


# ─── CLI 진입점 (standalone 테스트, LLM 호출 없음) ────────────────

if __name__ == "__main__":
    import argparse

    from app.preprocessing.utils import PROCESSED_DIR, label_from_csv, parse_and_split

    parser = argparse.ArgumentParser(description="항목 10: 비유/예시 활용 chunk 추출")
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
