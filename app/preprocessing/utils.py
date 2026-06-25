# Usage:
#   from app.preprocessing.utils import parse_and_split, label_from_csv
#
#   # 1) 텍스트 파일 파싱 + KSS 문장 분리 → data/processed/<stem>_kss.csv 저장
#   df = parse_and_split("data/raw/20260202_kdt-backendj-21th.txt")
#
#   # 2) CSV 발화 라벨링 (개념/예시/실습/기타) → <stem>_labeled.json 저장
#   labeled = label_from_csv("data/processed/2026-02-02_kdt-backendj-21th_kss.csv")

import asyncio
import json
import os
import re
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import kss
import pandas as pd
from dotenv import load_dotenv
from tqdm.auto import tqdm

load_dotenv()

_LLM_MODEL = os.environ.get("LLM_MODEL", "models/gemini-2.5-flash")
_LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.2"))

# google.generativeai 는 LLM 호출 시점에 지연 import·configure 한다.
# (KSS 문장 분리(split_sentences)만 쓰는 모듈이 genai/API_KEY 에 묶이지 않도록)
genai = None


def _ensure_genai():
    """google.generativeai 를 1회 import·configure 하고 모듈 핸들을 반환."""
    global genai
    if genai is None:
        import google.generativeai as _genai
        _genai.configure(api_key=os.environ["API_KEY"])
        genai = _genai
    return genai

LINE_RE = re.compile(r"^<(\d{2}:\d{2}:\d{2})>\s+(\S+):\s*(.*)$")

PROCESSED_DIR = Path("data/processed")

BEFORE_SEC    = 30
AFTER_SEC     = 210
BREAK_GAP_SEC = 600


# ── 파싱 + KSS ───────────────────────────────────────────────────────────────

def split_sentences(text: str, backend: str = "auto") -> list[str]:
    """KSS 문장 분리 (공유 입력 기준 단일 진입점, backend="auto")."""
    return kss.split_sentences(text, backend=backend)


def parse_and_split(txt_path: str | Path) -> pd.DataFrame:
    """텍스트 파일을 파싱하고 KSS로 문장을 분리한 뒤 data/processed/ 에 CSV로 저장한다.

    Args:
        txt_path: 강의 텍스트 파일 경로 (파일명 형식: YYYYMMDD_lecture-id.txt)

    Returns:
        columns: lecture_id, date, timestamp, speaker_id, text_raw
    """
    txt_path = Path(txt_path)
    date, *rest = txt_path.stem.split("_", 1)
    lecture_id = rest[0] if rest else txt_path.stem

    rows = []
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        m = LINE_RE.match(line.strip())
        if not m:
            continue
        rows.append({
            "lecture_id": lecture_id,
            "date": date,
            "timestamp": m.group(1),
            "speaker_id": m.group(2),
            "text_raw": m.group(3),
        })

    if not rows:
        return pd.DataFrame(columns=["lecture_id", "date", "timestamp", "speaker_id", "text_raw"])

    df = pd.DataFrame(_split_with_timestamps(pd.DataFrame(rows)))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / f"{txt_path.stem}_kss.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")

    return df


def _split_with_timestamps(group: pd.DataFrame) -> list[dict]:
    texts = group["text_raw"].tolist()
    timestamps = group["timestamp"].tolist()
    speakers = group["speaker_id"].tolist()

    joined = ""
    offsets = []
    for t in texts:
        offsets.append(len(joined))
        joined += t + " "
    joined = joined.rstrip()

    sentences = split_sentences(joined)

    results = []
    search_from = 0

    for sent in sentences:
        sent_start = joined.find(sent, search_from)
        if sent_start == -1:
            sent_start = search_from

        lo, hi = 0, len(offsets) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if offsets[mid] <= sent_start:
                lo = mid
            else:
                hi = mid - 1

        results.append({
            "lecture_id": group["lecture_id"].iloc[0],
            "date": group["date"].iloc[0],
            "timestamp": timestamps[lo],
            "speaker_id": speakers[lo],
            "text_raw": sent,
        })
        search_from = sent_start + len(sent)

    return results


# ── Step 1: Regex 라벨 탐지 ──────────────────────────────────────────────────

_CONCEPT_PATTERNS = [
    r"이라고 하는 것은", r"이라는 것은", r"이란 것은", r"이라는 뜻",
    r"이란 말이야", r"이라고 한다", r"이라고 합니다",
    r"이라고 얘기한다", r"이라고 얘기합니다",
    r"개념이", r"개념을", r"개념으로",
    r"원리가", r"원리는", r"특징이", r"특징은", r"특징을",
]

_EXAMPLE_PATTERNS = [
    r"예를 들어서", r"예를 들면", r"예를 들어", r"예를 들자면",
    r"예를들면", r"예를들어",
    r"비유하면", r"비유를 하면", r"비유를 들면", r"비유를 들자면",
    r"이를테면", r"가령",
]

_PRACTICE_PATTERNS = [
    r"해보세요", r"해봐요", r"해보십시오", r"해보도록", r"해볼게요", r"해봐",
    r"코드를 작성", r"코드 작성", r"작성해보", r"코딩해보", r"코딩하면", r"코딩하겠",
    r"실행해보", r"실행을 해보", r"실행해볼", r"한번 실행",
    r"복사해서", r"붙여보", r"붙여보세요", r"붙여보십시오",
    r"직접 해보", r"직접 해봐", r"따라 해보", r"따라해보", r"따라오",
    r"인텔리제이", r"이클립스", r"IDE",
]


def _compile(patterns: list[str]) -> re.Pattern:
    return re.compile("|".join(f"(?:{p})" for p in patterns))


_concept_re  = _compile(_CONCEPT_PATTERNS)
_example_re  = _compile(_EXAMPLE_PATTERNS)
_practice_re = _compile(_PRACTICE_PATTERNS)


def _detect_labels(text: str) -> list[str]:
    labels = []
    if _concept_re.search(text):  labels.append("개념")
    if _example_re.search(text):  labels.append("예시")
    if _practice_re.search(text): labels.append("실습")
    return labels if labels else ["기타"]


# ── Step 2: 앵커 창 병합 ──────────────────────────────────────────────────────

def _trim_at_break(window: pd.DataFrame, break_gap_sec: int = BREAK_GAP_SEC) -> pd.DataFrame:
    if window.empty:
        return window
    w = window.sort_values("dt").reset_index(drop=True)
    gaps = w["dt"].diff().dt.total_seconds()
    break_idxs = gaps[gaps >= break_gap_sec].index
    if break_idxs.empty:
        return w
    return w.iloc[:break_idxs[0]]


def _make_anchored_chunks(df: pd.DataFrame) -> pd.DataFrame:
    anchors = df[df["labels"].apply(lambda l: l != ["기타"])].copy()
    anchors = anchors.sort_values(["file", "dt"]).reset_index(drop=True)

    chunks = []
    for fname, grp in anchors.groupby("file", sort=False):
        grp = grp.sort_values("dt").reset_index(drop=True)
        i = 0
        while i < len(grp):
            cur = grp.iloc[i]
            end_dt = cur["dt"] + timedelta(seconds=AFTER_SEC)

            j = i + 1
            merged_labels = [cur["labels"]]
            while j < len(grp) and grp.iloc[j]["dt"] <= end_dt:
                end_dt = grp.iloc[j]["dt"] + timedelta(seconds=AFTER_SEC)
                merged_labels.append(grp.iloc[j]["labels"])
                j += 1

            t0 = cur["dt"] - timedelta(seconds=BEFORE_SEC)
            window = df[(df["file"] == fname) & (df["dt"] >= t0) & (df["dt"] <= end_dt)]
            window = _trim_at_break(window)

            all_labels = [l for ls in merged_labels for l in ls if l != "기타"]
            anchor_label = Counter(all_labels).most_common(1)[0][0] if all_labels else "기타"

            chunks.append({
                "file":         fname,
                "anchor_dt":    cur["dt"],
                "anchor_label": anchor_label,
                "anchor_text":  cur["text"],
                "n_merged":     j - i,
                "n_utterances": len(window),
                "text":         " ".join(window["text"].tolist()),
            })
            i = j

    return pd.DataFrame(chunks)


# ── Step 3: Gemini 최종 라벨 ──────────────────────────────────────────────────

def _make_prompt(anchor_text: str, chunk_text: str) -> str:
    return f"""다음은 한국어 강의 전사 텍스트입니다.
키워드가 탐지된 핵심 발화(ANCHOR)와 전후 약 3분 맥락(CONTEXT)을 함께 제공합니다.

[ANCHOR 발화]
{anchor_text}

[전후 맥락]
{chunk_text}

분류 기준:
- 개념: 텍스트에서 특정 용어·원리·특징을 **명시적으로** 정의하거나 설명하는 문장이 있어야 함
- 예시: "예를 들어", "비유하면" 등 표현과 함께 구체적 사례가 **실제로** 제시되어야 함
- 실습: 코드 작성·실행·IDE 조작 등 수행 지시가 **명확하게** 있어야 함
- 기타: 위 기준을 확실히 충족하는 증거가 없으면 **무조건 기타**

⚠️ 판단 지침:
- 애매하거나 억지로 끼워 맞춰야 한다면 → 기타
- 개념처럼 보여도 정의 문장이 명시적으로 없다면 → 기타
- 실습 키워드가 있어도 주변 맥락이 잡담이라면 → 기타

아래 JSON 형식으로만 응답하세요:
{{
  "label": "개념 또는 예시 또는 실습 또는 기타",
  "reason": "한 줄 판단 근거 (기타인 경우 증거가 없는 이유)",
  "key_sentence": "개념인 경우 텍스트에서 그대로 인용한 정의 문장, 아니면 null"
}}"""


async def _classify_one(
    model, semaphore: asyncio.Semaphore, pbar, idx: int, row: pd.Series
) -> dict:
    async with semaphore:
        try:
            response = await asyncio.to_thread(
                model.generate_content,
                _make_prompt(row["anchor_text"], row["text"]),
                generation_config=genai.GenerationConfig(response_mime_type="application/json"),
            )
            result = json.loads(response.text)
        except Exception as e:
            result = {"label": "ERROR", "reason": str(e), "key_sentence": None}
        finally:
            pbar.update(1)
        return {"idx": idx, **result}


async def _run_classification(chunks_df: pd.DataFrame, concurrency: int = 15) -> pd.DataFrame:
    _ensure_genai()
    model = genai.GenerativeModel(
        _LLM_MODEL,
        generation_config=genai.GenerationConfig(temperature=_LLM_TEMPERATURE),
    )
    semaphore = asyncio.Semaphore(concurrency)
    pbar = tqdm(total=len(chunks_df), desc="Gemini 라벨링 중")
    tasks = [_classify_one(model, semaphore, pbar, idx, row) for idx, row in chunks_df.iterrows()]
    results = await asyncio.gather(*tasks)
    pbar.close()

    res_df = pd.DataFrame(results).set_index("idx")
    chunks_df = chunks_df.copy()
    chunks_df["llm_label"]        = res_df["label"]
    chunks_df["llm_reason"]       = res_df["reason"]
    chunks_df["llm_key_sentence"] = res_df["key_sentence"]
    return chunks_df


# ── 라벨링 파이프라인 ─────────────────────────────────────────────────────────

_LABEL_COLS = [
    "file", "anchor_dt", "anchor_label", "n_merged",
    "anchor_text", "n_utterances", "text",
    "llm_label", "llm_reason", "llm_key_sentence",
]


def label_from_csv(csv_path: str | Path, concurrency: int = 15) -> list[dict]:
    """CSV 파일 전체를 읽어 라벨링한 뒤 data/processed/ 에 JSON으로 저장한다.

    Args:
        csv_path: parse_and_split()으로 생성된 CSV 경로 (단일 강의)
        concurrency: Gemini 동시 요청 수

    Returns:
        컬럼: file, anchor_dt, anchor_label, anchor_text, n_merged,
               n_utterances, text, llm_label, llm_reason, llm_key_sentence
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    result = asyncio.run(_label(df, concurrency=concurrency))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / f"{csv_path.stem.removesuffix('_kss')}_labeled.json"
    result[_LABEL_COLS].to_json(
        output_path, orient="records", indent=2, force_ascii=False, date_format="iso"
    )

    return json.loads(output_path.read_text(encoding="utf-8"))


async def _label(df: pd.DataFrame, concurrency: int = 15) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["text_raw"] if "text_raw" in df.columns else df["text"]
    df["dt"]   = pd.to_datetime(df["date"] + " " + df["timestamp"])
    df["file"] = df["date"] + "_" + df["lecture_id"] + ".txt"
    df = df.sort_values(["file", "dt"]).reset_index(drop=True)

    df["labels"] = df["text"].apply(_detect_labels)

    chunks_df = _make_anchored_chunks(df)
    chunks_df = await _run_classification(chunks_df, concurrency)
    return chunks_df