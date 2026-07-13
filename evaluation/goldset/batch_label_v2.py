"""
개념/예시/실습 v2 분절 배치 실행 (사람 검수 전 LLM 초안)
대상 날짜: DATES 리스트 참고
출력: data/goldset/labeling/{date}_concept_goldset_v2.json
       data/goldset/labeling/{date}_example_goldset_v2.json
       data/goldset/labeling/{date}_practice_goldset_v2.json
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd
import google.generativeai as genai
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.core.config import settings

genai.configure(api_key=settings.api_key)
MODEL = settings.eval_model

CSV_PATH = Path(__file__).parent.parent / "data/processed/lectures_kss.csv"
GS = Path(__file__).parent.parent / "data/goldset/labeling"
GS.mkdir(parents=True, exist_ok=True)

DATES = ["2026-02-10", "2026-02-11", "2026-02-12", "2026-02-13"]

WINDOW = 120
STRIDE = 100


# ── 윈도우 포매터 ────────────────────────────────────────────────────

def _format_window(sentences: list[str], start: int, end: int) -> str:
    return "\n".join(f"[{i}] {sentences[i]}" for i in range(start, end))


# ── 개념 분절 ────────────────────────────────────────────────────────

SEG_SYSTEM = """당신은 강의 전사 텍스트에서 '구별되는 개념'을 식별하고
각 개념이 설명되는 구간을 하나의 덩어리로 묶는 전문가입니다.
응답은 반드시 JSON만 출력하세요."""

SEG_USER = """아래는 강의 전사의 한 구간입니다. 각 줄은 [전역인덱스] 문장 형식입니다.

이 구간에서 강사가 명시적으로 정의하거나 설명하는 '구별되는 개념'을 식별하세요.

【묶기 규칙 — 가장 중요】
- 하나의 개념이 여러 문장에 걸쳐 설명되면, 그 전체를 하나의 구간(start_index~end_index)으로 묶습니다.
- 중간에 짧은 곁가지(예시 한두 마디 등)가 껴도 같은 개념 설명이 이어지면 하나로 봅니다.
- 서로 다른 개념은 각각 별도 항목으로 분리합니다. (개념 1개 = 항목 1개)

【제외】
- 예시 들기, 실습 지시("해보세요"), 진행 안내, 잡담은 개념이 아닙니다.
- 개념을 '정의·설명'하지 않고 단순 언급만 하는 문장도 제외합니다.

각 개념에 대해:
- concept_name: 개념의 핵심 이름
- start_index / end_index: 시작/끝 문장의 전역 인덱스 (반드시 아래 구간 내 값)
- key_sentence: 그 개념을 가장 잘 정의·설명하는 핵심 문장 원문 1개

문장 구간:
{indexed_sentences}

아래 JSON 형식으로만 응답하세요:
{{"concepts": [{{"concept_name": "...", "start_index": 0, "end_index": 0, "key_sentence": "..."}}]}}"""


def _parse_concepts(raw: str) -> list[dict]:
    text = re.sub(r"^```\w*\s*", "", raw.strip())
    text = re.sub(r"\s*```$", "", text).strip()
    try:
        return json.loads(text).get("concepts", [])
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        return json.loads(m.group()).get("concepts", []) if m else []


def run_segmentation(sentences: list[str]) -> list[dict]:
    model = genai.GenerativeModel(MODEL)
    out: list[dict] = []
    for ws in tqdm(range(0, len(sentences), STRIDE), desc="concept"):
        we = min(ws + WINDOW, len(sentences))
        prompt = f"{SEG_SYSTEM}\n\n{SEG_USER.format(indexed_sentences=_format_window(sentences, ws, we))}"
        resp = model.generate_content(prompt, generation_config=genai.GenerationConfig(
            response_mime_type="application/json", temperature=settings.llm_temperature))
        for c in _parse_concepts(resp.text):
            try:
                s, e = int(c["start_index"]), int(c["end_index"])
            except (KeyError, ValueError, TypeError):
                continue
            if s > e:
                s, e = e, s
            s, e = max(s, ws), min(e, we - 1)
            if s > e:
                continue
            out.append({"concept_name": str(c.get("concept_name", "")).strip(),
                        "start": s, "end": e,
                        "key_sentence": str(c.get("key_sentence", "")).strip()})
        if we >= len(sentences):
            break
    return out


def dedup_concepts(items: list[dict]) -> list[dict]:
    merged: list[dict] = []
    for c in sorted(items, key=lambda x: (x["start"], x["end"])):
        if merged and c["start"] <= merged[-1]["end"]:
            prev = merged[-1]
            prev["end"] = max(prev["end"], c["end"])
            if len(c["concept_name"]) > len(prev["concept_name"]):
                prev["concept_name"] = c["concept_name"]
        else:
            merged.append(dict(c))
    return merged


# ── 예시 분절 ────────────────────────────────────────────────────────

EX_SYSTEM = """당신은 강의 전사 텍스트에서 '예시(example)'를 식별하고
각 예시가 제시되는 구간을 하나의 덩어리로 묶는 전문가입니다.
응답은 반드시 JSON만 출력하세요."""

EX_USER = """아래는 강의 전사의 한 구간입니다. 각 줄은 [전역인덱스] 문장 형식입니다.

이 구간에서 강사가 **구체적인 값·이름·숫자·상황을 들어 함수/개념의 동작이나 결과를
보여주는 '예시'** 를 식별하세요.

【포함】
- "예를 들어 ~", "만약에 X를 넣으면 Y가 된다" 처럼 구체 입력→결과를 보여주는 구간
- 특정 숫자·이름·문자로 동작을 시연하는 구간

【묶기 규칙 — 가장 중요】
- 하나의 예시가 여러 문장에 걸치면 그 전체를 하나의 구간(start_index~end_index)으로 묶습니다.
- 서로 다른 예시는 각각 별도 항목으로 분리합니다. (예시 1개 = 항목 1개)

【제외】
- 구체 값 없이 일반적으로 정의·설명만 하는 문장(= 개념)
- 수강생 독립 과제(= 실습), 진행 안내·잡담

각 예시에 대해:
- example_name: 무엇을 보여주는 예시인지 짧은 이름
- start_index / end_index: 시작/끝 문장의 전역 인덱스 (반드시 구간 내 값)
- key_sentence: 핵심 문장 원문 1개

문장 구간:
{indexed_sentences}

아래 JSON 형식으로만 응답하세요:
{{"examples": [{{"example_name": "...", "start_index": 0, "end_index": 0, "key_sentence": "..."}}]}}"""


def _parse_examples(raw: str) -> list[dict]:
    text = re.sub(r"^```\w*\s*", "", raw.strip())
    text = re.sub(r"\s*```$", "", text).strip()
    try:
        return json.loads(text).get("examples", [])
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        return json.loads(m.group()).get("examples", []) if m else []


def run_example_segmentation(sentences: list[str]) -> list[dict]:
    model = genai.GenerativeModel(MODEL)
    out: list[dict] = []
    for ws in tqdm(range(0, len(sentences), STRIDE), desc="example"):
        we = min(ws + WINDOW, len(sentences))
        prompt = f"{EX_SYSTEM}\n\n{EX_USER.format(indexed_sentences=_format_window(sentences, ws, we))}"
        resp = model.generate_content(prompt, generation_config=genai.GenerationConfig(
            response_mime_type="application/json", temperature=settings.llm_temperature))
        for c in _parse_examples(resp.text):
            try:
                s, e = int(c["start_index"]), int(c["end_index"])
            except (KeyError, ValueError, TypeError):
                continue
            if s > e:
                s, e = e, s
            s, e = max(s, ws), min(e, we - 1)
            if s > e:
                continue
            out.append({"example_name": str(c.get("example_name", "")).strip(),
                        "start": s, "end": e,
                        "key_sentence": str(c.get("key_sentence", "")).strip()})
        if we >= len(sentences):
            break
    return out


def dedup_examples(items: list[dict]) -> list[dict]:
    merged: list[dict] = []
    for c in sorted(items, key=lambda x: (x["start"], x["end"])):
        if merged and c["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], c["end"])
            if len(c["example_name"]) > len(merged[-1]["example_name"]):
                merged[-1]["example_name"] = c["example_name"]
        else:
            merged.append(dict(c))
    return merged


# ── 실습 분절 ────────────────────────────────────────────────────────

PR_SYSTEM = """당신은 강의 전사 텍스트에서 '실습 과제'를 식별하고
각 과제가 진행되는 구간을 하나의 덩어리로 묶는 전문가입니다.
응답은 반드시 JSON만 출력하세요."""

PR_USER = """아래는 강의 전사의 한 구간입니다. 각 줄은 [전역인덱스] 문장 형식입니다.

이 구간에서 **수강생이 독립적으로 완성해야 하는 실습 과제**를 식별하세요.

【포함】
- 번호가 있는 문제: "Q1 풀어보세요", "1번 구현해보자"
- 처음부터 완성해야 하는 SQL 과제 (테이블 생성·데이터 입력·쿼리 작성)
- 강사가 "이걸 해보세요"라고 독립 과제를 제시하고 풀이가 이어지는 구간

【묶기 규칙 — 가장 중요】
- 하나의 과제(문제 제시 + 풀이 진행)가 여러 문장에 걸치면 그 전체를 하나의 구간으로 묶습니다.
- 서로 다른 과제는 각각 별도 항목으로 분리합니다. (과제 1개 = 항목 1개)

【제외】
- 강사 설명/시범 중 즉흥적 따라하기("이렇게 해봐", "한번 써봐")
- 단순 실행·클릭·값 입력, 개념 정의·예시 시연만 하는 구간

각 과제에 대해:
- practice_name: 무슨 과제인지 짧은 이름
- start_index / end_index: 시작/끝 문장의 전역 인덱스 (반드시 구간 내 값)
- key_sentence: 과제를 가장 잘 나타내는 핵심 문장 원문 1개

문장 구간:
{indexed_sentences}

아래 JSON 형식으로만 응답하세요:
{{"practices": [{{"practice_name": "...", "start_index": 0, "end_index": 0, "key_sentence": "..."}}]}}"""


def _parse_practices(raw: str) -> list[dict]:
    text = re.sub(r"^```\w*\s*", "", raw.strip())
    text = re.sub(r"\s*```$", "", text).strip()
    try:
        return json.loads(text).get("practices", [])
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        return json.loads(m.group()).get("practices", []) if m else []


def run_practice_segmentation(sentences: list[str]) -> list[dict]:
    model = genai.GenerativeModel(MODEL)
    out: list[dict] = []
    for ws in tqdm(range(0, len(sentences), STRIDE), desc="practice"):
        we = min(ws + WINDOW, len(sentences))
        prompt = f"{PR_SYSTEM}\n\n{PR_USER.format(indexed_sentences=_format_window(sentences, ws, we))}"
        resp = model.generate_content(prompt, generation_config=genai.GenerationConfig(
            response_mime_type="application/json", temperature=settings.llm_temperature))
        for c in _parse_practices(resp.text):
            try:
                s, e = int(c["start_index"]), int(c["end_index"])
            except (KeyError, ValueError, TypeError):
                continue
            if s > e:
                s, e = e, s
            s, e = max(s, ws), min(e, we - 1)
            if s > e:
                continue
            out.append({"practice_name": str(c.get("practice_name", "")).strip(),
                        "start": s, "end": e,
                        "key_sentence": str(c.get("key_sentence", "")).strip()})
        if we >= len(sentences):
            break
    return out


def dedup_practices(items: list[dict]) -> list[dict]:
    merged: list[dict] = []
    for c in sorted(items, key=lambda x: (x["start"], x["end"])):
        if merged and c["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], c["end"])
            if len(c["practice_name"]) > len(merged[-1]["practice_name"]):
                merged[-1]["practice_name"] = c["practice_name"]
        else:
            merged.append(dict(c))
    return merged


# ── 날짜별 실행 ──────────────────────────────────────────────────────

def run_for_date(date: str, df_all: pd.DataFrame) -> None:
    print(f"\n{'='*60}")
    print(f"날짜: {date}")

    df = df_all[df_all["date"] == date].reset_index(drop=True)
    if df.empty:
        print(f"  [건너뜀] 데이터 없음")
        return

    sentences = df["text_raw"].fillna("").tolist()
    ts = df["timestamp"].tolist()
    print(f"  문장 수: {len(sentences)}")

    def enrich(items: list[dict], name_key: str) -> list[dict]:
        for c in items:
            c["text"] = " ".join(sentences[i] for i in range(c["start"], c["end"] + 1))
            c["timestamp"] = ts[c["start"]]
        return items

    # 개념
    concepts_raw = run_segmentation(sentences)
    concepts = enrich(dedup_concepts(concepts_raw), "concept_name")
    co_path = GS / f"{date}_concept_goldset_v2.json"
    co_path.write_text(json.dumps(
        {"source": CSV_PATH.name, "date": date, "method": "llm_segmentation_v2",
         "concept_count": len(concepts), "concepts": concepts},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  개념 {len(concepts_raw)}개 → dedup → {len(concepts)}개  →  {co_path.name}")

    # 예시
    examples_raw = run_example_segmentation(sentences)
    examples = enrich(dedup_examples(examples_raw), "example_name")
    ex_path = GS / f"{date}_example_goldset_v2.json"
    ex_path.write_text(json.dumps(
        {"source": CSV_PATH.name, "date": date, "method": "llm_segmentation_v2",
         "example_count": len(examples), "examples": examples},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  예시 {len(examples_raw)}개 → dedup → {len(examples)}개  →  {ex_path.name}")

    # 실습
    practices_raw = run_practice_segmentation(sentences)
    practices = enrich(dedup_practices(practices_raw), "practice_name")
    pr_path = GS / f"{date}_practice_goldset_v2.json"
    pr_path.write_text(json.dumps(
        {"source": CSV_PATH.name, "date": date, "method": "llm_segmentation_v2",
         "practice_count": len(practices), "practices": practices},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  실습 {len(practices_raw)}개 → dedup → {len(practices)}개  →  {pr_path.name}")


# ── main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df_all = pd.read_csv(CSV_PATH)
    for date in DATES:
        run_for_date(date, df_all)
    print("\n\n완료. 각 날짜별 _v2.json 파일을 사람이 검수해 _final.json 을 만드세요.")
