"""
app/analysis/item15_error_handling.py — 항목 15 오류 대응 평가 (Step 2·3·4)

담당자: 박다현

흐름: preprocessing.item15_error_handling.run(df) → practice, candidates
      → Step 2: Gemini yes/no 검증 (실제 오류 대응인가?)
      → Step 3: Gemini A/B/C 품질 판단 (원인 / 단계적 해결 / 확인)
      → Step 4: 청크별 1~5점 채점 → 파일별 평균 = 최종 점수

입력: 단일 강의 kss DataFrame
출력: {"item_id": 15, "summary": dict, "file_scores": dict, "chunk_results": list}
"""

from __future__ import annotations

import asyncio
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from pathlib import Path as _Path

import yaml as _yaml

from app.core.config import settings
from app.preprocessing.item15_error_handling import run as build_candidates

_LLM_MODEL = settings.llm_model
_LLM_TEMPERATURE = settings.llm_temperature

# Gemini 신 SDK 클라이언트 (지연 초기화)
_client = None


def _get_client():
    global _client
    if _client is None:
        from google import genai

        _client = genai.Client(api_key=settings.api_key)
    return _client

MODULE_NAME = "error_handling"
ITEM_ID = 15
_STEM = "item15_error_handling"

_PROMPTS_DIR = _Path(__file__).resolve().parent / "prompts" / "items"

def _load_prompt() -> dict:
    with open(_PROMPTS_DIR / f"{_STEM}.yaml", encoding="utf-8") as f:
        return _yaml.safe_load(f)

_PROMPT: dict = _load_prompt()


async def _verify_one(client, sem: asyncio.Semaphore, idx: int, row: dict) -> dict:
    from google.genai import types

    matched = ", ".join(f"'{h['matched']}'" for h in row["hits"][:5])
    verify = _PROMPT["verify"]
    prompt = (
        verify["user_template"]
        .replace("{matched}", matched)
        .replace("{anchor_text}", row["anchor_text"])
        .replace("{text}", row["text"])
    )
    async with sem:
        try:
            resp = await client.aio.models.generate_content(
                model=_LLM_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=_LLM_TEMPERATURE,
                    system_instruction=verify["system"],
                ),
            )
            raw = resp.text.strip()
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            parsed = json.loads(m.group()) if m else {}
            return {
                "idx": idx,
                "llm_is_error":     parsed.get("is_error_handling"),
                "llm_reason":       parsed.get("reason"),
                "llm_key_sentence": parsed.get("key_sentence"),
            }
        except Exception as e:
            return {"idx": idx, "llm_is_error": None, "llm_reason": str(e), "llm_key_sentence": None}


# ── Step 3: Gemini A/B/C 판단 ──────────────────────────────────────────────

def _make_abc_prompt(criterion: str, row: dict) -> str:
    matched = ", ".join(f"'{h['matched']}'" for h in row["hits"][:5]) or "(없음)"
    key = row.get("llm_key_sentence") or "(없음)"
    crit = _PROMPT["criteria"][criterion]
    user = (
        crit["user_template"]
        .replace("{matched}", matched)
        .replace("{key}", key)
        .replace("{text}", row["text"][:2000])
    )
    return f"{crit['system']}\n\n{user}"


async def _eval_criterion(client, criterion: str, row: dict) -> dict:
    from google.genai import types

    prompt = _make_abc_prompt(criterion, row)
    try:
        resp = await client.aio.models.generate_content(
            model=_LLM_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=_LLM_TEMPERATURE,
                response_mime_type="application/json",
            ),
        )
        parsed = json.loads(resp.text)
        return {"result": parsed.get("result"), "evidence": parsed.get("evidence")}
    except Exception as e:
        return {"result": None, "evidence": str(e)}


async def _eval_abc_one(client, sem: asyncio.Semaphore, idx: int, row: dict) -> dict:
    async with sem:
        a, b, c = await asyncio.gather(
            _eval_criterion(client, "A", row),
            _eval_criterion(client, "B", row),
            _eval_criterion(client, "C", row),
        )
    return {
        "idx": idx,
        "abc_A": a["result"], "abc_A_evidence": a["evidence"],
        "abc_B": b["result"], "abc_B_evidence": b["evidence"],
        "abc_C": c["result"], "abc_C_evidence": c["evidence"],
    }


# ── Step 4: 채점 ────────────────────────────────────────────────────────────

def _score_chunk(A: bool, B: str, C: bool) -> int:
    """루브릭:
    5: A✓ B=full C✓
    4: A✓ B=full C✗
    3: A✗ B=full  (C 무관)
    2: B=partial
    1: B=none
    """
    if B == "full":
        if A and C:
            return 5
        if A:
            return 4
        return 3
    if B == "partial":
        return 2
    return 1


# ── 메인 파이프라인 ─────────────────────────────────────────────────────────

async def run(
    df: pd.DataFrame,
    concurrency: int = 10,
) -> dict:
    """오류 대응 평가 파이프라인 실행 (Step 2·3·4).

    Args:
        df: 하루치 강의 발화 DataFrame (단일 date).
        concurrency: Gemini 동시 요청 수.

    Returns:
        JSON-직렬화 가능한 결과 dict.
    """
    preprocessed = build_candidates(df)
    practice: list[dict] = preprocessed["practice"]
    candidates: list[dict] = preprocessed["candidates"]

    client = _get_client()
    sem = asyncio.Semaphore(concurrency)

    # ── Step 2: Gemini yes/no ──────────────────────────────────
    verify_results = await asyncio.gather(
        *[_verify_one(client, sem, i, row) for i, row in enumerate(candidates)]
    )
    verify_map = {r["idx"]: r for r in verify_results}

    for i, row in enumerate(candidates):
        v = verify_map[i]
        row["llm_is_error"]     = v["llm_is_error"]
        row["llm_reason"]       = v["llm_reason"]
        row["llm_key_sentence"] = v["llm_key_sentence"]

    true_rows = [r for r in candidates if r.get("llm_is_error") is True]

    # ── Step 3: Gemini A/B/C ──────────────────────────────────
    abc_results = await asyncio.gather(
        *[_eval_abc_one(client, sem, i, row) for i, row in enumerate(true_rows)]
    )
    abc_map = {r["idx"]: r for r in abc_results}

    for i, row in enumerate(true_rows):
        a = abc_map[i]
        row.update({
            "abc_A": a["abc_A"], "abc_A_evidence": a["abc_A_evidence"],
            "abc_B": a["abc_B"], "abc_B_evidence": a["abc_B_evidence"],
            "abc_C": a["abc_C"], "abc_C_evidence": a["abc_C_evidence"],
        })

    # ── Step 4: 채점 + 파일별 집계 ────────────────────────────
    for row in true_rows:
        row["score"] = _score_chunk(row["abc_A"], row["abc_B"], row["abc_C"])

    file_groups: dict[str, list[dict]] = defaultdict(list)
    for row in true_rows:
        file_groups[row["file"]].append(row)

    file_scores: dict[str, dict] = {}
    for file, rows in file_groups.items():
        scores = [r["score"] for r in rows]
        dist = {str(s): sum(1 for x in scores if x == s) for s in range(1, 6)}
        file_scores[file] = {
            "final_score":        round(sum(scores) / len(scores)),
            "n_error_chunks":     len(scores),
            "score_min":          min(scores),
            "score_max":          max(scores),
            "score_distribution": dist,
        }

    all_final = [v["final_score"] for v in file_scores.values()]
    overall = round(sum(all_final) / len(all_final)) if all_final else None

    chunk_results = [
        {
            "file":             r["file"],
            "anchor_dt":        r["anchor_dt"],
            "anchor_text":      r["anchor_text"],
            "first_match":      r["first_match"],
            "llm_is_error":     r["llm_is_error"],
            "llm_key_sentence": r["llm_key_sentence"],
            "abc_A":            r["abc_A"],
            "abc_A_evidence":   r["abc_A_evidence"],
            "abc_B":            r["abc_B"],
            "abc_B_evidence":   r["abc_B_evidence"],
            "abc_C":            r["abc_C"],
            "abc_C_evidence":   r["abc_C_evidence"],
            "score":            r["score"],
        }
        for r in true_rows
    ]

    return {
        "item_id":      15,
        "item_name":    "오류 대응",
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "final_score":  overall,
        "summary": {
            "total_practice_chunks": len(practice),
            "regex_hit_chunks":      len(candidates),
            "regex_hit_rate":        round(len(candidates) / len(practice), 3) if practice else 0,
            "verified_error_chunks": len(true_rows),
            "llm_precision":         round(len(true_rows) / len(candidates), 3) if candidates else 0,
            "overall_final_score":   overall,
        },
        "file_scores":   file_scores,
        "chunk_results": chunk_results,
    }


# ── CLI 진입점 ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    from app.preprocessing.utils import PROCESSED_DIR, label_from_csv, parse_and_split

    parser = argparse.ArgumentParser(description="항목 15: 오류 대응 평가 (Step 2·3·4)")
    parser.add_argument("txt_path", help="원본 강의 텍스트 파일 경로 (단일 강의)")
    parser.add_argument("--output", "-o", default=None, help="결과 JSON 저장 경로 (생략 시 stdout)")
    parser.add_argument("--concurrency", "-c", type=int, default=10, help="Gemini 동시 요청 수 (기본 10)")
    args = parser.parse_args()

    _txt = Path(args.txt_path)
    parse_and_split(_txt)
    _csv = PROCESSED_DIR / f"{_txt.stem}_kss.csv"
    _labeled = pd.DataFrame(label_from_csv(_csv, concurrency=args.concurrency))
    result = asyncio.run(run(_labeled, concurrency=args.concurrency))

    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"저장 완료: {args.output}")
    else:
        print(output)
