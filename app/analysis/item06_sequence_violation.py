"""
app/analysis/item06_sequence_violation.py — 항목 6 예시 순서 위반 탐지 (Step 3·4·5)

흐름: preprocessing.item06_sequence_violation.run(df) → seq_df, concept_blocks
      → Step 3: 위반 A/B 후보 탐지
      → Step 4: 위반 B 후보 Gemini 예시 필요 여부 판단
      → Step 5: 위반 건수 집계 + 점수 산정

입력: 단일 강의 kss DataFrame
출력: {"final_score": int, "violation_counts": dict, "topic_pattern_dist": dict, "meta": dict}
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from app.analysis import rubric_llm
from app.preprocessing.item06_sequence_violation import KW_THRESHOLD
from app.preprocessing.item06_sequence_violation import run as build_data

MODULE_NAME = "sequence_violation"
ITEM_ID = 6
_STEM = "item06_sequence_violation"


# ── Step 3: 위반 A/B 후보 탐지 ───────────────────────────────────────────────

def _extract_violations(
    concept_blocks: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    viol_a_rows: list[dict] = []
    viol_b_cands: list[dict] = []
    topic_groups: list[dict] = []

    for block in concept_blocks:
        base = {k: block[k] for k in ("file", "dt", "text", "llm_key_sentence")}
        same_topic: pd.DataFrame = block["same_topic"]
        window: pd.DataFrame = block["window"]
        n_ex: int = block["n_예시"]
        n_prac: int = block["n_실습"]
        pat: str = block["pattern"]
        seq_str: str = block["seq_str"]

        topic_groups.append({**base, "pattern": pat, "n_예시": n_ex, "n_실습": n_prac})

        if seq_str == "개념만":
            viol_b_cands.append({**base, "seq_str": seq_str, "next_text": "", "next_label": "다음 구간"})
            continue

        examples = same_topic[same_topic["llm_label"] == "예시"]
        practices = same_topic[same_topic["llm_label"] == "실습"]

        if n_ex == 0:
            first_prac = practices.iloc[0] if n_prac > 0 else window.iloc[0]
            viol_b_cands.append({
                **base,
                "seq_str": seq_str,
                "next_text": first_prac["text"],
                "next_label": first_prac["llm_label"],
            })
        elif n_prac > 0:
            first_ex_dt = examples["anchor_dt"].min()
            if len(practices[practices["anchor_dt"] < first_ex_dt]) > 0:
                viol_a_rows.append({**base, "seq_str": seq_str})

    return pd.DataFrame(viol_a_rows), pd.DataFrame(viol_b_cands), pd.DataFrame(topic_groups)


# ── Step 4: Gemini 예시 필요 여부 판단 ───────────────────────────────────────

def _build_chunk(concept_text: str, key_sentence: str, next_text: str, next_label: str) -> str:
    return (
        f"[개념 설명 — 핵심 정의 문장]\n{key_sentence}\n\n"
        f"[개념 설명 — 전체 맥락]\n{concept_text[:1200]}\n\n"
        f"[다음 구간 ({next_label})]\n{str(next_text)[:600]}\n\n"
        f"---\n\n"
        f"이 개념 설명 다음에 별도의 예시 없이 바로 {next_label}으로 넘어갔습니다.\n"
        f"개념 자체의 성질과 이어지는 {next_label} 내용을 함께 고려합니다."
    )


async def _classify_one(
    prompt: dict, semaphore: asyncio.Semaphore, pbar, idx: int, row: pd.Series
) -> dict:
    chunk = _build_chunk(
        row["text"],
        row.get("llm_key_sentence") or "",
        row.get("next_text") or "",
        row.get("next_label") or "다음 구간",
    )
    result = await rubric_llm.judge(
        prompt["system"], rubric_llm.render_user(prompt, chunk), semaphore
    )
    if "_error" in result:
        result = {"needs_example": None, "reason": result["_error"]}
    pbar.update(1)
    return {"idx": idx, **result}


async def _run_llm_judgment(cands_df: pd.DataFrame, concurrency: int = 15) -> pd.DataFrame:
    if cands_df.empty:
        cands_df = cands_df.copy()
        cands_df["needs_example"] = pd.Series(dtype=object)
        cands_df["skip_reason"] = pd.Series(dtype=object)
        return cands_df

    prompt = rubric_llm.load_prompt(_STEM)
    sem = asyncio.Semaphore(concurrency)
    pbar = tqdm(total=len(cands_df), desc="예시 필요 여부 판단 중")
    try:
        tasks = [_classify_one(prompt, sem, pbar, idx, row) for idx, row in cands_df.iterrows()]
        results = await asyncio.gather(*tasks)
    finally:
        pbar.close()

    res_df = pd.DataFrame(results).set_index("idx")
    cands_df = cands_df.copy()
    cands_df["needs_example"] = res_df["needs_example"]
    cands_df["skip_reason"] = res_df["reason"]
    return cands_df


# ── Step 5: 점수 산정 ─────────────────────────────────────────────────────────

def _calc_score(seq_df: pd.DataFrame, total_violations: int) -> int:
    present = set(seq_df["llm_label"].unique())
    if not {"개념", "실습", "예시"}.issubset(present):
        return 1
    if total_violations >= 6:
        return 1
    if total_violations <= 3:
        return {0: 5, 1: 4, 2: 3, 3: 3}[total_violations]
    return 2  # 4–5


# ── 메인 파이프라인 ───────────────────────────────────────────────────────────

async def run(
    df: pd.DataFrame,
    concurrency: int = 15,
    threshold: float = KW_THRESHOLD,
) -> dict:
    """순서 위반 탐지 파이프라인 실행 (Step 3·4·5).

    Args:
        df: 하루치 레이블된 강의 청크 DataFrame (단일 date).
            필수 컬럼: llm_label, text (또는 text_raw)
            선택 컬럼: file, anchor_dt, llm_key_sentence
        concurrency: Gemini 동시 요청 수.
        threshold: 키워드 overlap 임계값.

    Returns:
        JSON-직렬화 가능한 결과 dict.
    """
    preprocessed = build_data(df, threshold=threshold)
    seq_df: pd.DataFrame = preprocessed["seq_df"]
    concept_blocks: list[dict] = preprocessed["concept_blocks"]

    viol_a_df, viol_b_cands_df, topic_df = _extract_violations(concept_blocks)
    viol_b_cands_df = await _run_llm_judgment(viol_b_cands_df, concurrency)
    viol_b_df = viol_b_cands_df[viol_b_cands_df["needs_example"] == True].copy()

    viol_a_count = int(len(viol_a_df))
    viol_b_count = int(len(viol_b_df))
    total = viol_a_count + viol_b_count

    return {
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "final_score": _calc_score(seq_df, total),
        "violation_counts": {
            "위반A": viol_a_count,
            "위반B": viol_b_count,
            "합계": total,
        },
        "topic_pattern_dist": topic_df["pattern"].value_counts().to_dict(),
        "meta": {
            "total_chunks": int(len(df)),
            "concept_blocks": int(len(topic_df)),
            "viol_b_candidates": int(len(viol_b_cands_df)),
            "kw_threshold": threshold,
        },
    }


# ── CLI 진입점 ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    from app.preprocessing.utils import PROCESSED_DIR, label_from_csv, parse_and_split

    parser = argparse.ArgumentParser(description="항목 6: 예시 순서 위반 탐지 (Step 3·4·5)")
    parser.add_argument("txt_path", help="원본 강의 텍스트 파일 경로 (단일 강의)")
    parser.add_argument("--output", "-o", default=None, help="결과 JSON 저장 경로 (생략 시 stdout)")
    parser.add_argument("--concurrency", "-c", type=int, default=15, help="Gemini 동시 요청 수 (기본 15)")
    parser.add_argument(
        "--threshold", "-t", type=float, default=KW_THRESHOLD,
        help=f"키워드 overlap 임계값 (기본 {KW_THRESHOLD})",
    )
    args = parser.parse_args()

    _txt = Path(args.txt_path)
    parse_and_split(_txt)
    _csv = PROCESSED_DIR / f"{_txt.stem}_kss.csv"
    _labeled = pd.DataFrame(label_from_csv(_csv, concurrency=args.concurrency))
    result = asyncio.run(run(_labeled, concurrency=args.concurrency, threshold=args.threshold))

    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"저장 완료: {args.output}")
    else:
        print(output)
