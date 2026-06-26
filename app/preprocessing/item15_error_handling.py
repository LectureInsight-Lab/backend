"""항목 15: 오류 대응 평가 — 전처리 (Step 1)

담당자: 박다현

Pipeline 전체:
    실습 청크 (llm_label == "실습")
      → Step 1 : Regex — 오류 대응 후보 탐지   ← 여기까지
      → Step 2~4 : app/analysis/item15_error_handling 참조
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

# ── Step 1: Regex 패턴 ──────────────────────────────────────────────────────

_DIRECT = [
    r"에러가?\s*(났|나|떴|뜨|잡|있|생)",
    r"오류가?\s*(났|나|떴|뜨|잡|있|생)",
    r"에러\s*(메세지|메시지|코드|내용|보면|확인)",
    r"오류\s*(메세지|메시지|코드|내용|보면|확인)",
    r"(런타임|컴파일|문법)\s*(에러|오류)",
    r"워닝\s*(났|나|떴|뜨|잡|있)",
    r"(경고|warning)\s*(났|나|떴|뜨|있)",
    r"빨간\s*줄",
    r"빨간색\s*(표시|밑줄|선)",
    r"null\s*pointer",
    r"exception\s*(났|나|떴|잡)",
]
_SYMPTOM = [
    r"안\s*나와(요|서|도)?",
    r"안\s*떠(요|서|도)?",
    r"안\s*실행(돼|되)",
    r"실행\s*안\s*(돼|되)",
    r"결과가?\s*(안\s*나|안\s*뜨|틀려|이상해)",
    r"막혔(어요?|는데)",
    r"안\s*되는(데|거|건)",
    r"안\s*(돼요|됩니다|되는\s*거)",
    r"왜\s*(이러|그러|안\s*되)",
    r"(이게|이거)\s*문제",
    r"거기서\s*막히",
]
_FIX = [
    r"(이렇게\s*하면|그렇게\s*하면)\s*안\s*(돼|됩니|되는)",
    r"(여기|이\s*부분|이거)\s*(수정|고쳐|바꿔)",
    r"(수정|고쳐|다시\s*봐|다시\s*확인)(해|봐|야|하면|하세요)",
    r"(왜\s*안\s*되냐|왜\s*에러|왜\s*오류)",
    r"(오류\s*잡|에러\s*잡|오류를\s*고|에러를\s*고)",
    r"다시\s*(해보|실행|확인)(해|봐|요)?",
]
ALL_PATTERNS: list[str] = _DIRECT + _SYMPTOM + _FIX
_ERROR_RE = re.compile("|".join(ALL_PATTERNS), re.IGNORECASE)


def _find_hits(text: str) -> list[dict]:
    hits = []
    for pattern in ALL_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            hits.append({
                "matched": m.group(),
                "pattern": pattern,
                "context": text[max(0, m.start() - 60): m.end() + 60],
            })
    return sorted(hits, key=lambda x: x["matched"])


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "text" not in df.columns and "text_raw" in df.columns:
        df["text"] = df["text_raw"]
    if "file" not in df.columns:
        df["file"] = df.get("date", df.get("lecture_id", "unknown"))
    if "anchor_dt" not in df.columns:
        df["anchor_dt"] = df.get("timestamp", "")
    if "anchor_text" not in df.columns:
        df["anchor_text"] = df["text"].str[:80]
    return df


def run(df: pd.DataFrame) -> dict:
    """Step 1 실행: Regex 오류 대응 후보 탐지.

    Args:
        df: 하루치 강의 발화 DataFrame.
            필수 컬럼: text (또는 text_raw)
            선택 컬럼: llm_label, file, anchor_dt

    Returns:
        {
            "practice": list[dict],    # 실습 청크 전체
            "candidates": list[dict],  # regex 히트 후보
        }
    """
    df = _normalize(df)
    chunks = df.to_dict(orient="records")

    if "llm_label" in df.columns:
        practice = [c for c in chunks if c.get("llm_label") == "실습"]
    else:
        practice = chunks

    candidates = []
    for c in practice:
        hits = _find_hits(c["text"])
        if hits:
            candidates.append({
                "file":        c["file"],
                "anchor_dt":   c["anchor_dt"],
                "anchor_text": c["anchor_text"],
                "text":        c["text"],
                "hits":        hits,
                "first_match": hits[0]["matched"],
            })

    return {"practice": practice, "candidates": candidates}


# ── CLI 진입점 ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    from app.preprocessing.utils import PROCESSED_DIR, label_from_csv, parse_and_split

    parser = argparse.ArgumentParser(description="항목 15: 오류 대응 Regex 후보 탐지 (Step 1)")
    parser.add_argument("txt_path", help="원본 강의 텍스트 파일 경로 (단일 강의)")
    parser.add_argument("--output", "-o", default=None, help="결과 JSON 저장 경로 (생략 시 stdout)")
    parser.add_argument("--concurrency", "-c", type=int, default=10, help="레이블링 동시 요청 수 (기본 10)")
    args = parser.parse_args()

    _txt = Path(args.txt_path)
    parse_and_split(_txt)
    _csv = PROCESSED_DIR / f"{_txt.stem}_kss.csv"
    _labeled = pd.DataFrame(label_from_csv(_csv, concurrency=args.concurrency))
    result = run(_labeled)

    summary = {
        "total_practice_chunks": len(result["practice"]),
        "regex_hit_chunks": len(result["candidates"]),
        "candidates": [
            {"file": c["file"], "first_match": c["first_match"], "n_hits": len(c["hits"])}
            for c in result["candidates"]
        ],
    }
    output = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"저장 완료: {args.output}")
    else:
        print(output)
