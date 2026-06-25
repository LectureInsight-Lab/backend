"""항목 15: 오류 대응 평가 파이프라인.

Pipeline:
    실습 청크 (llm_label == "실습")
      → Step 1 : Regex  — 오류 대응 후보 탐지
      → Step 2 : Gemini — yes/no 검증 (실제 오류 대응인가?)
      → Step 3 : Gemini — A/B/C 품질 판단 (원인 / 단계적 해결 / 확인)
      → Step 4 : 청크별 1~5점 채점 → 파일별 평균 = 최종 점수
      → Output : JSON

Usage:
    import asyncio, json
    from app.analysis.error_handling import run

    result = asyncio.run(run(chunks))      # chunks: anchored_labeled.json 목록
    print(json.dumps(result, ensure_ascii=False, indent=2))
"""

from __future__ import annotations

import asyncio
import json
import pandas as pd
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import google.generativeai as genai

from app.core.config import settings

genai.configure(api_key=settings.api_key)

_LLM_MODEL = settings.llm_model
_LLM_TEMPERATURE = settings.llm_temperature

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


# ── Step 2: Gemini yes/no 검증 ──────────────────────────────────────────────

_SYSTEM_YES_NO = """당신은 강의 품질 평가 전문가입니다.
주어진 실습 구간 텍스트가 강사의 **오류 대응** 행동을 포함하는지 판단합니다.

[오류 대응 정의]
강사가 학생이 코드 실행·작성 중 만난 에러·오류·경고를 인식하고,
원인을 설명하거나 수정 방법을 안내하거나, 올바른 방향으로 유도하는 발화.

[오류 대응 해당 예]
- "여기 빨간 줄 나오죠? 타입이 안 맞아서 그래요. 이렇게 바꾸면 돼요."
- "오류 메시지 보면 NullPointerException이라고 나오죠. 초기화가 안 된 거예요."
- "워닝 났죠? 이건 에러는 아닌데 나중에 문제 생길 수 있으니 수정해봐요."

[해당하지 않는 예]
- "에러 처리를 try-catch로 한다"  (개념 설명)
- "이렇게 하면 안 돼"             (단순 코드 방향 설명)
- "안 나와도 돼, 그냥 넘어가자"   (오류 무시)

반드시 JSON으로만 응답:
{"is_error_handling": true 또는 false, "reason": "판단 근거 1~2문장", "key_sentence": "핵심 근거 문장 또는 null"}"""


async def _verify_one(model, sem: asyncio.Semaphore, idx: int, row: dict) -> dict:
    matched = ", ".join(f"'{h['matched']}'" for h in row["hits"][:5])
    prompt = (
        f"[Regex 탐지 표현]\n{matched}\n\n"
        f"[ANCHOR 발화]\n{row['anchor_text']}\n\n"
        f"[실습 구간 맥락]\n{row['text']}\n\n"
        "위 텍스트에 강사의 오류 대응 행동이 포함되어 있습니까? JSON으로 답하세요."
    )
    async with sem:
        try:
            resp = await asyncio.to_thread(
                model.generate_content, [_SYSTEM_YES_NO, prompt]
            )
            raw = resp.text.strip()
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            parsed = json.loads(m.group()) if m else {}
            return {
                "idx": idx,
                "llm_is_error": parsed.get("is_error_handling"),
                "llm_reason": parsed.get("reason"),
                "llm_key_sentence": parsed.get("key_sentence"),
            }
        except Exception as e:
            return {"idx": idx, "llm_is_error": None, "llm_reason": str(e), "llm_key_sentence": None}


# ── Step 3: Gemini A/B/C 판단 ──────────────────────────────────────────────

_CRITERION_PROMPTS = {
    "A": """다음은 한국어 실습 강의 전사 텍스트입니다.

아래 **한 가지 기준만** 판단하세요:
▶ 강사가 오류/에러의 **원인(왜 발생했는지)**을 설명하는 발화가 텍스트에 있는가?

판단 지침:
- 단순히 오류가 났다는 언급만으로는 부족 — 원인 설명이 명시적으로 있어야 함
- "이 타입이 맞지 않아서", "초기화가 안 돼서", "여기가 빠져서" 같은 표현이 있으면 true
- 오류 언급 없이 그냥 코드 설명이면 false

JSON으로만 응답: {"result": true 또는 false, "evidence": "근거 문장 또는 null"}""",

    "B": """다음은 한국어 실습 강의 전사 텍스트입니다.

아래 **한 가지 기준만** 판단하세요:
▶ 강사가 오류 해결 방법을 어느 수준으로 안내했는가?

세 가지 중 하나로 판단:
- "full"   : 해결 **과정**이 담긴 안내 — 어디를 보고, 무엇을 확인하고, 어떻게 고치는지 흐름을 알 수 있음
             예) "SHOW WARNINGS 보면 어떤 에러인지 나와 → 거기서 원인 잡으면 돼"
             예) "서버 변수 수정하면 서비스 껐다 켜야 돼, 안 그러면 반영이 안 돼"
             예) "딜리트 하고 오류 메시지 보면 대상이 없다 이러니까 → FROM 줘야 되는 거야"
             핵심: 단순 정답 제시가 아니라, 학생이 과정을 이해하고 따라할 수 있는 수준
- "partial": 정답/결과만 제시 — 무엇을 하면 되는지만 알려주고 과정·흐름 없이 끝냄
             예) "없으면 빼면 되지 뭐", "이렇게 하면 됩니다", "비밀번호 넣어주면 돼요"
             핵심: what(정답)은 있지만 how/why(과정)가 없는 단답형
- "none"   : 해결 시도 자체가 없는 경우
             예) 오류 언급만 하고 넘어감, "나중에 보자", 오류 무시

주의: A(원인 설명)와 B는 독립적입니다.
      원인을 설명했어도 해결 안내가 단답형이면 B="partial"입니다.
      단, "어디서 확인 → 어떻게 고침"처럼 진단-해결 흐름이 이어지면 B="full"입니다.

JSON으로만 응답: {"result": "full" 또는 "partial" 또는 "none", "evidence": "근거 문장 또는 null"}""",

    "C": """다음은 한국어 실습 강의 전사 텍스트입니다.

아래 **한 가지 기준만** 판단하세요:
▶ 강사가 오류 해결 후 학생이 잘 됐는지 **확인하는 발화**가 텍스트에 있는가?

판단 지침:
- "됐죠?", "잘 되셨나요?", "확인해보세요", "잘 나와요?", "해결됐나요?" 같은 표현이 있으면 true
- 오류 대응 후 별도 확인 없이 다음 주제로 넘어가면 false
- 강사 혼자 실행하며 "됐다" 확인하는 것도 true

JSON으로만 응답: {"result": true 또는 false, "evidence": "근거 문장 또는 null"}""",
}


def _make_abc_prompt(criterion: str, row: dict) -> str:
    matched = ", ".join(f"'{h['matched']}'" for h in row["hits"][:5]) or "(없음)"
    key = row.get("llm_key_sentence") or "(없음)"
    return (
        f"{_CRITERION_PROMPTS[criterion]}\n\n"
        f"[Regex 탐지 표현]: {matched}\n"
        f"[핵심 오류 대응 문장]: {key}\n\n"
        f"[실습 구간 맥락]\n{row['text'][:2000]}"
    )


async def _eval_criterion(model, criterion: str, row: dict) -> dict:
    prompt = _make_abc_prompt(criterion, row)
    try:
        resp = await asyncio.to_thread(
            model.generate_content,
            prompt,
            generation_config=genai.GenerationConfig(response_mime_type="application/json"),
        )
        parsed = json.loads(resp.text)
        return {"result": parsed.get("result"), "evidence": parsed.get("evidence")}
    except Exception as e:
        return {"result": None, "evidence": str(e)}


async def _eval_abc_one(model, sem: asyncio.Semaphore, idx: int, row: dict) -> dict:
    async with sem:
        a, b, c = await asyncio.gather(
            _eval_criterion(model, "A", row),
            _eval_criterion(model, "B", row),
            _eval_criterion(model, "C", row),
        )
    return {
        "idx": idx,
        "abc_A": a["result"], "abc_A_evidence": a["evidence"],
        "abc_B": b["result"], "abc_B_evidence": b["evidence"],
        "abc_C": c["result"], "abc_C_evidence": c["evidence"],
    }


# ── Step 4: 채점 ────────────────────────────────────────────────────────────

def _score_chunk(A: bool, B: str, C: bool) -> int:
    """루빅:
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

def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """lectures_kss.csv 컬럼을 파이프라인 내부 컬럼명으로 통일."""
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


async def run(
    df: pd.DataFrame,
    concurrency: int = 10,
) -> dict:
    """오류 대응 평가 파이프라인 실행.

    Args:
        df: 하루치 강의 발화 DataFrame (단일 date).
            lectures_kss.csv 컬럼 기준: date, timestamp, speaker_id, text_raw
            llm_label 컬럼이 없으면 전체 발화에 regex 적용.
        concurrency: Gemini 동시 요청 수.

    Returns:
        JSON-직렬화 가능한 결과 dict.
    """
    df = _normalize(df)
    chunks = df.to_dict(orient="records")
    model = genai.GenerativeModel(
        _LLM_MODEL,
        generation_config=genai.GenerationConfig(temperature=_LLM_TEMPERATURE),
    )
    sem = asyncio.Semaphore(concurrency)

    # ── Step 1: Regex ──────────────────────────────────────────
    if "llm_label" in df.columns:
        practice = [c for c in chunks if c.get("llm_label") == "실습"]
    else:
        practice = chunks
    candidates = []
    for c in practice:
        hits = _find_hits(c["text"])
        if hits:
            candidates.append({
                "file":         c["file"],
                "anchor_dt":    c["anchor_dt"],
                "anchor_text":  c["anchor_text"],
                "text":         c["text"],
                "hits":         hits,
                "first_match":  hits[0]["matched"],
            })

    # ── Step 2: Gemini yes/no ──────────────────────────────────
    tasks2 = [_verify_one(model, sem, i, row) for i, row in enumerate(candidates)]
    verify_results = await asyncio.gather(*tasks2)
    verify_map = {r["idx"]: r for r in verify_results}

    for i, row in enumerate(candidates):
        v = verify_map[i]
        row["llm_is_error"]     = v["llm_is_error"]
        row["llm_reason"]       = v["llm_reason"]
        row["llm_key_sentence"] = v["llm_key_sentence"]

    true_rows = [r for r in candidates if r.get("llm_is_error") is True]

    # ── Step 3: Gemini A/B/C ──────────────────────────────────
    tasks3 = [_eval_abc_one(model, sem, i, row) for i, row in enumerate(true_rows)]
    abc_results = await asyncio.gather(*tasks3)
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
            "final_score":       round(sum(scores) / len(scores)),
            "n_error_chunks":    len(scores),
            "score_min":         min(scores),
            "score_max":         max(scores),
            "score_distribution": dist,
        }

    all_final = [v["final_score"] for v in file_scores.values()]
    overall = round(sum(all_final) / len(all_final), 2) if all_final else None

    # ── 청크 결과 직렬화 ──────────────────────────────────────
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
        "item_id":   15,
        "item_name": "오류 대응",
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "summary": {
            "total_practice_chunks":  len(practice),
            "regex_hit_chunks":       len(candidates),
            "regex_hit_rate":         round(len(candidates) / len(practice), 3) if practice else 0,
            "verified_error_chunks":  len(true_rows),
            "llm_precision":          round(len(true_rows) / len(candidates), 3) if candidates else 0,
            "overall_final_score":    overall,
        },
        "file_scores":    file_scores,
        "chunk_results":  chunk_results,
    }


# ── CLI 진입점 ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    from app.preprocessing.utils import PROCESSED_DIR, label_from_csv, parse_and_split

    parser = argparse.ArgumentParser(description="항목 15 오류 대응 평가")
    parser.add_argument("txt_path", help="원본 강의 텍스트 파일 경로 (단일 강의)")
    parser.add_argument("--output", "-o", default=None,  help="결과 JSON 저장 경로 (생략 시 stdout)")
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
