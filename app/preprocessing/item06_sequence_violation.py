"""항목 6: 예시 순서 위반 탐지 — 전처리 (Step 1·2)

담당자: 박다현

Pipeline 전체:
    강의 1개 CSV
      → Step 1 : 기타 제외, 시간순 정렬 + 명사 키워드 추출 (kiwipiepy)
      → Step 2 : 개념 블록별 동일 주제 판별 (overlap coefficient ≥ 0.1)   ← 여기까지
      → Step 3~5 : app/analysis/item06_sequence_violation 참조

Usage:
    python -m app.preprocessing.item06_sequence_violation lecture.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from kiwipiepy import Kiwi

KW_THRESHOLD = 0.1

_kiwi = Kiwi()
STOPWORDS = {
    "다음", "경우", "질문", "설명", "내용", "부분", "얘기", "이야기",
    "생각", "문제", "방법", "기본", "처음", "마지막", "단계", "순서",
    "사람", "분들", "여러분", "선생님", "강사", "학생", "구현", "키워드",
}


# ── Step 1: 전처리 + 키워드 추출 ─────────────────────────────────────────────

def _extract_keywords(text: str) -> set[str]:
    tokens = _kiwi.tokenize(text[:500])
    return {
        t.form for t in tokens
        if t.tag in ("NNG", "NNP", "SL") and len(t.form) >= 2 and t.form not in STOPWORDS
    }


def _overlap_coef(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "text" not in df.columns and "text_raw" in df.columns:
        df["text"] = df["text_raw"]
    if "file" not in df.columns:
        df["file"] = df.get("date", df.get("lecture_id", "unknown"))
    if "anchor_dt" not in df.columns:
        df["anchor_dt"] = df.get("timestamp", "")
    return df


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize(df)
    df["anchor_dt"] = pd.to_datetime(df["anchor_dt"])
    seq_df = (
        df[df["llm_label"] != "기타"]
        .sort_values(["file", "anchor_dt"])
        .reset_index(drop=True)
    )
    seq_df["keywords"] = seq_df["text"].apply(_extract_keywords)
    return seq_df


# ── Step 2: 개념 블록별 동일 주제 판별 ───────────────────────────────────────

def _build_concept_blocks(
    seq_df: pd.DataFrame, threshold: float = KW_THRESHOLD
) -> list[dict]:
    """개념 블록별 동일 주제 판별 — overlap coefficient ≥ threshold.

    Returns:
        list of dicts, each representing one concept block:
            file, dt, text, llm_key_sentence,
            same_topic (DataFrame), window (DataFrame),
            n_예시, n_실습, pattern, seq_str
    """
    concept_blocks: list[dict] = []

    for file, grp in seq_df.groupby("file"):
        grp = grp.sort_values("anchor_dt").reset_index(drop=True)
        concept_idxs = grp.index[grp["llm_label"] == "개념"].tolist()

        for k, ci in enumerate(concept_idxs):
            concept_row = grp.iloc[ci]
            concept_kw = concept_row["keywords"]
            end_idx = concept_idxs[k + 1] if k + 1 < len(concept_idxs) else len(grp)
            window = grp.iloc[ci + 1:end_idx].copy()
            prac_ex = window[window["llm_label"].isin(["실습", "예시"])].copy()

            base = {
                "file": file,
                "dt": concept_row["anchor_dt"],
                "text": concept_row["text"],
                "llm_key_sentence": concept_row.get("llm_key_sentence", ""),
            }

            if len(prac_ex) == 0:
                concept_blocks.append({
                    **base,
                    "same_topic": pd.DataFrame(),
                    "window": window,
                    "n_예시": 0,
                    "n_실습": 0,
                    "pattern": "개념만",
                    "seq_str": "개념만",
                })
                continue

            prac_ex["overlap"] = prac_ex["keywords"].apply(
                lambda kw: _overlap_coef(concept_kw, kw)
            )
            same_topic = prac_ex[prac_ex["overlap"] >= threshold].sort_values("anchor_dt")
            examples = same_topic[same_topic["llm_label"] == "예시"]
            practices = same_topic[same_topic["llm_label"] == "실습"]
            n_ex, n_prac = len(examples), len(practices)

            if n_ex > 0 and n_prac > 0:
                pat = "개념+예시+실습"
            elif n_ex > 0:
                pat = "개념+예시"
            elif n_prac > 0:
                pat = "개념+실습"
            else:
                pat = "개념만"

            seq_str = " → ".join(same_topic["llm_label"].tolist()) if len(same_topic) else "동일주제없음"

            concept_blocks.append({
                **base,
                "same_topic": same_topic,
                "window": window,
                "n_예시": n_ex,
                "n_실습": n_prac,
                "pattern": pat,
                "seq_str": seq_str,
            })

    return concept_blocks


def run(df: pd.DataFrame, threshold: float = KW_THRESHOLD) -> dict:
    """Steps 1–2 실행: 전처리 + 개념 블록별 동일 주제 판별.

    Args:
        df: 하루치 레이블된 강의 청크 DataFrame.
            필수 컬럼: llm_label, text (또는 text_raw)
            선택 컬럼: file, anchor_dt, llm_key_sentence
        threshold: 키워드 overlap 임계값.

    Returns:
        {
            "seq_df": pd.DataFrame,        # Step 1 결과
            "concept_blocks": list[dict],  # Step 2 결과
        }
    """
    seq_df = _prepare(df)
    concept_blocks = _build_concept_blocks(seq_df, threshold)
    return {"seq_df": seq_df, "concept_blocks": concept_blocks}


# ── CLI 진입점 ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    from app.preprocessing.utils import PROCESSED_DIR, label_from_csv, parse_and_split

    parser = argparse.ArgumentParser(description="항목 6: 예시 순서 위반 전처리 (Step 1·2)")
    parser.add_argument("txt_path", help="원본 강의 텍스트 파일 경로 (단일 강의)")
    parser.add_argument("--output", "-o", default=None, help="결과 JSON 저장 경로 (생략 시 stdout)")
    parser.add_argument("--concurrency", "-c", type=int, default=15, help="레이블링 동시 요청 수 (기본 15)")
    parser.add_argument(
        "--threshold", "-t", type=float, default=KW_THRESHOLD,
        help=f"키워드 overlap 임계값 (기본 {KW_THRESHOLD})",
    )
    args = parser.parse_args()

    _txt = Path(args.txt_path)
    parse_and_split(_txt)
    _csv = PROCESSED_DIR / f"{_txt.stem}_kss.csv"
    _labeled = pd.DataFrame(label_from_csv(_csv, concurrency=args.concurrency))
    result = run(_labeled, threshold=args.threshold)

    summary = {
        "total_chunks": int(len(result["seq_df"])),
        "concept_blocks": len(result["concept_blocks"]),
        "pattern_dist": pd.DataFrame(result["concept_blocks"])["pattern"].value_counts().to_dict()
        if result["concept_blocks"] else {},
    }
    output = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"저장 완료: {args.output}")
    else:
        print(output)
