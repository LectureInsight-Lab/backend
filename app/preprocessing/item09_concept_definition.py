"""
app/preprocessing/item09_concept_definition.py — 항목 9 개념 정의

담당자: 심소민

라벨링된 청크(labeled) 중 llm_label == '개념' 청크를 추출해 정의 평가용 chunk 목록으로
반환한다. 청크가 1500자 미만이면 인접 청크와 병합해 컨텍스트를 확보한다.
개념명 추출·정규화·중복 제거·정의 채점은 app/analysis(LLM)가 담당한다.

흐름: run(df) → '개념' 청크 추출 → <1500자 인접 병합 → chunk 목록
입력: 단일 강의 labeled DataFrame (label_from_csv 결과). 필수 컬럼: llm_label, text
출력: {"chunks": [{"request_id": str, "text": str, "context": {...}}, ...]}
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

MODULE_NAME = "concept_definition"
ITEM_ID = 9

MIN_CHUNK_CHARS = 1500  # 미만이면 인접 청크와 병합


# ─── 전처리 → chunk 생성 ──────────────────────────────────────────

def _concept_chunks(labeled: pd.DataFrame) -> list[dict]:
    """llm_label == '개념' 청크를 추출하고 1500자 미만이면 인접 청크와 병합한다."""
    concept_chunks = labeled[labeled["llm_label"] == "개념"].reset_index(drop=True)

    result: list[dict] = []
    for i, row in concept_chunks.iterrows():
        text = str(row.get("text", ""))
        if len(text) < MIN_CHUNK_CHARS:
            prev_text = str(concept_chunks.iloc[i - 1]["text"]) if i > 0 else ""
            next_text = str(concept_chunks.iloc[i + 1]["text"]) if i + 1 < len(concept_chunks) else ""
            text = f"{prev_text}\n{text}\n{next_text}".strip()

        result.append({
            "request_id": f"concept_{i}",
            "text": text,
            "context": {"source": str(row.get("file", ""))},
        })

    return result


def run(df: pd.DataFrame) -> dict:
    """단일 강의 labeled DataFrame → 개념 정의 평가용 chunk 목록.

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

    parser = argparse.ArgumentParser(description="항목 9: 개념 정의 chunk 추출")
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
