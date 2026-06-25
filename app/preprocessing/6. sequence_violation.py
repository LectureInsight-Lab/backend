"""항목: 예시 순서 위반 탐지 파이프라인 (키워드 겹침 기반).

Pipeline:
    강의 1개 CSV
      → Step 1 : 기타 제외, 시간순 정렬 + 명사 키워드 추출 (kiwipiepy)
      → Step 2 : 개념 블록별 동일 주제 판별 (overlap coefficient ≥ 0.1)
      → Step 3 : 위반 A/B 후보 탐지
          위반 A : 동일 주제 내 실습이 예시보다 먼저 등장
          위반 B 후보 : 동일 주제 내 예시 없음
      → Step 4 : 위반 B 후보 → Gemini 예시 필요 여부 판단
      → Step 5 : 위반 건수 집계 + 개념 블록 패턴 분포

Usage:
    python sequence_violation.py --input lecture.csv
    python sequence_violation.py --input lecture.csv --output result.json

    또는 모듈로:
        import asyncio, pandas as pd
        from app.analysis.sequence_violation import run

        df = pd.read_csv("lecture.csv")
        result = asyncio.run(run(df))
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

import google.generativeai as genai
import pandas as pd
from dotenv import load_dotenv
from kiwipiepy import Kiwi
from tqdm.auto import tqdm

load_dotenv()
genai.configure(api_key=os.environ["API_KEY"])

_LLM_MODEL = os.environ.get("LLM_MODEL", "models/gemini-2.5-flash")
_LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.2"))
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


# ── Step 2–3: 키워드 기반 위반 탐지 ──────────────────────────────────────────

def _detect_violations(
    seq_df: pd.DataFrame, threshold: float = KW_THRESHOLD
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    viol_a_rows: list[dict] = []
    viol_b_cands: list[dict] = []
    topic_groups: list[dict] = []

    for file, grp in seq_df.groupby("file"):
        grp = grp.sort_values("anchor_dt").reset_index(drop=True)
        concept_idxs = grp.index[grp["llm_label"] == "개념"].tolist()

        for k, ci in enumerate(concept_idxs):
            concept_row = grp.iloc[ci]
            concept_kw = concept_row["keywords"]
            end_idx = concept_idxs[k + 1] if k + 1 < len(concept_idxs) else len(grp)
            window = grp.iloc[ci + 1:end_idx]
            prac_ex = window[window["llm_label"].isin(["실습", "예시"])].copy()

            base = {
                "file": file,
                "dt": concept_row["anchor_dt"],
                "text": concept_row["text"],
                "llm_key_sentence": concept_row.get("llm_key_sentence", ""),
            }

            if len(prac_ex) == 0:
                topic_groups.append({**base, "pattern": "개념만", "n_예시": 0, "n_실습": 0})
                viol_b_cands.append({**base, "seq_str": "개념만", "next_text": "", "next_label": "다음 구간"})
                continue

            prac_ex["overlap"] = prac_ex["keywords"].apply(lambda kw: _overlap_coef(concept_kw, kw))
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
            topic_groups.append({**base, "pattern": pat, "n_예시": n_ex, "n_실습": n_prac})

            seq_str = " → ".join(same_topic["llm_label"].tolist()) if len(same_topic) else "동일주제없음"

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

def _make_skip_prompt(concept_text: str, key_sentence: str, next_text: str, next_label: str) -> str:
    return f"""당신은 백엔드 입문 강의를 평가하는 교육 전문가입니다.

아래는 강의 전사 텍스트에서 개념 설명으로 분류된 구간과, 그 바로 다음에 이어지는 구간입니다.

[개념 설명 — 핵심 정의 문장]
{key_sentence}

[개념 설명 — 전체 맥락]
{concept_text[:1200]}

[다음 구간 ({next_label})]
{str(next_text)[:600]}

---

이 개념 설명 다음에 별도의 예시 없이 바로 {next_label}으로 넘어갔습니다.
이 흐름에서 **예시가 없어도 학습자가 개념을 이해할 수 있는지** 판단하세요.
개념 자체의 성질과 이어지는 {next_label} 내용을 함께 고려합니다.

**예시가 필요한 개념 (needs_example: true)**
- 동작 원리·메커니즘 개념이라 "이게 실제로 어떻게 작동하는지"를 보여줘야 이해됨
- 여러 요소가 얽힌 구조라서 전체 흐름을 한 번 보여줘야 감이 잡히는 개념
- 정의만으로는 "언제, 어떤 상황에서 쓰는지"가 불분명하고, 이어지는 {next_label}에서도 그 맥락이 채워지지 않음

**예시가 필요하지 않은 개념 (needs_example: false)**
- 단순 명칭·용어 소개로 정의 자체가 직관적임
- 앞서 배운 개념의 소폭 확장이거나 맥락상 자연스럽게 따라오는 개념
- 정의 자체에 사용 방법이 내포되어 있어 추가 예시 없이도 적용 가능한 개념
- 이어지는 {next_label} 내용을 통해 이 개념이 어떻게 쓰이는지 자연스럽게 드러남

애매한 경우 false로 판단하세요.

아래 JSON 형식으로만 응답하세요:
{{
  "needs_example": true 또는 false,
  "reason": "판단 근거 — 개념의 성질 또는 이어지는 흐름 중 어느 쪽 기준인지 한 줄로"
}}"""


async def _classify_one(
    model, semaphore: asyncio.Semaphore, pbar, idx: int, row: pd.Series
) -> dict:
    async with semaphore:
        try:
            response = await asyncio.to_thread(
                model.generate_content,
                _make_skip_prompt(
                    row["text"],
                    row.get("llm_key_sentence") or "",
                    row.get("next_text") or "",
                    row.get("next_label") or "다음 구간",
                ),
                generation_config=genai.GenerationConfig(response_mime_type="application/json"),
            )
            result = json.loads(response.text)
        except Exception as e:
            result = {"needs_example": None, "reason": str(e)}
        finally:
            pbar.update(1)
        return {"idx": idx, **result}


async def _run_llm_judgment(cands_df: pd.DataFrame, concurrency: int = 15) -> pd.DataFrame:
    if cands_df.empty:
        cands_df = cands_df.copy()
        cands_df["needs_example"] = pd.Series(dtype=object)
        cands_df["skip_reason"] = pd.Series(dtype=object)
        return cands_df

    model = genai.GenerativeModel(
        _LLM_MODEL,
        generation_config=genai.GenerationConfig(temperature=_LLM_TEMPERATURE),
    )
    semaphore = asyncio.Semaphore(concurrency)
    pbar = tqdm(total=len(cands_df), desc="예시 필요 여부 판단 중")
    try:
        tasks = [_classify_one(model, semaphore, pbar, idx, row) for idx, row in cands_df.iterrows()]
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
    """순서 위반 탐지 파이프라인 실행.

    Args:
        df: 하루치 레이블된 강의 청크 DataFrame (단일 date).
            필수 컬럼: llm_label, text (또는 text_raw)
            선택 컬럼: file, anchor_dt (없으면 date/timestamp로 대체), llm_key_sentence
        concurrency: Gemini 동시 요청 수.
        threshold: 키워드 overlap 임계값.

    Returns:
        JSON-직렬화 가능한 결과 dict.
    """
    seq_df = _prepare(df)
    viol_a_df, viol_b_cands_df, topic_df = _detect_violations(seq_df, threshold)
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

    parser = argparse.ArgumentParser(description="예시 순서 위반 탐지 (키워드 겹침 기반)")
    parser.add_argument("txt_path", help="원본 강의 텍스트 파일 경로 (단일 강의)")
    parser.add_argument("--output", "-o", default=None,  help="결과 JSON 저장 경로 (생략 시 stdout)")
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
