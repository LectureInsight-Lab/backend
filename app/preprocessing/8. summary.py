"""수업 마무리 요약 컨텍스트 추출.

summary_eval_eda.ipynb 에서 확정된 로직 이식.
LectureDocument.outro_lines 를 받아 LLM에 넘길 JSON 페이로드를 반환한다.

트리거 탐지 우선순위:
    1. summary_signal  — 교사가 명시적으로 요약을 선언한 지점 앞 200자부터
    2. strong_end      — 수업 종료 발화 앞 800자부터
    3. weak_end        — 쉬는 시간 발화 앞 800자부터
    4. fallback        — 아무 신호 없을 때 maxi 마지막 3000자
"""

import json
import re
from pathlib import Path

import pandas as pd

from app.analysis.schemas import Utterance

# ─── 탐지 패턴 ──────────────────────────────────────────────────

_SUMMARY_PATTERNS = [
    r"정리하자면",
    r"정리하면",
    r"정리해\s*보면",
    r"정리\s*해보면",
    r"한번\s*정리",
    r"정리를\s*살짝",
    r"요약하자면",
    r"요약하면",
    r"오늘\s*(배운|배웠던|다룬|했던)\s*(내용|것|부분)",
    r"이번\s*시간에\s*(배운|다룬|했던)",
]

_STRONG_END_PATTERNS = [
    r"오늘은\s*여기까지",
    r"오늘은\s*여기까.{0,3}",
    r"여기까지\s*하겠습니다",
    r"여기까지\s*할게요",
    r"여기까지\s*하도록",
    r"수업\s*마치겠습니다",
    r"마치도록\s*하겠습니다",
    r"고생하셨습니다",
    r"고생\s*많으셨습니다",
    r"수고하셨습니다",
    r"수고\s*많으셨습니다",
    r"내일\s*보겠습니다",
    r"다음\s*시간에",
    r"내일은.{0,80}(진행|학습|보겠습니다|하겠습니다)",
    r"다음에는.{0,80}(진행|학습|보겠습니다|하겠습니다)",
]

_WEAK_END_PATTERNS = [
    r"쉬었다가",
    r"\d+\s*분\s*쉬",
]


def _last_match(text: str, patterns: list[str]) -> dict | None:
    """패턴 목록 중 가장 마지막에 등장하는 매치 반환."""
    matches = [
        {
            "start": m.start(),
            "matched_text": m.group(),
            "matched_pattern": p,
        }
        for p in patterns
        for m in re.finditer(p, text)
    ]
    return max(matches, key=lambda x: x["start"]) if matches else None


def build_summary_llm_payload(
    outro_lines: list[Utterance],
    prev_chars_summary: int = 200,
    prev_chars_end: int = 800,
    fallback_chars: int = 4000,
    max_context_chars: int = 4000,
) -> dict:
    """outro_lines → LLM에 넘길 JSON 페이로드.

    Args:
        outro_lines:        LectureDocument.outro_lines (수업 종료 구간 발화)
        prev_chars_summary: summary 트리거 기준 앞으로 볼 문자 수
        prev_chars_end:     strong/weak_end 트리거 기준 앞으로 볼 문자 수
        fallback_chars:     트리거 없을 때 마지막 N자
        max_context_chars:  LLM에 넘길 최대 문자 수

    Returns:
        {
            "context":         str,   # LLM이 읽을 텍스트
            "extraction_type": str,   # summary_signal | strong_end | weak_end | fallback
            "trigger":         str,   # 감지된 트리거 발화 (없으면 "")
        }
    """
    section_text = re.sub(r"\s+", " ", " ".join(u.text for u in outro_lines)).strip()

    summary_match   = _last_match(section_text, _SUMMARY_PATTERNS)
    strong_end_match = _last_match(section_text, _STRONG_END_PATTERNS)
    weak_end_match  = _last_match(section_text, _WEAK_END_PATTERNS)

    if summary_match:
        start = max(0, summary_match["start"] - prev_chars_summary)
        extraction_type = "summary_signal"
        trigger = summary_match["matched_text"]
    elif strong_end_match:
        start = max(0, strong_end_match["start"] - prev_chars_end)
        extraction_type = "strong_end"
        trigger = strong_end_match["matched_text"]
    elif weak_end_match:
        start = max(0, weak_end_match["start"] - prev_chars_end)
        extraction_type = "weak_end"
        trigger = weak_end_match["matched_text"]
    else:
        start = max(0, len(section_text) - fallback_chars)
        extraction_type = "fallback"
        trigger = ""

    context = section_text[start:]
    if len(context) > max_context_chars:
        context = context[:max_context_chars]

    return {
        "context": context,
        "extraction_type": extraction_type,
        "trigger": trigger,
    }


OUTRO_MINUTES = 15


def _extract_outro(df: pd.DataFrame) -> pd.DataFrame:
    """timestamp 기준으로 마지막 OUTRO_MINUTES분 발화만 반환."""
    df = df.copy()
    df["_ts"] = pd.to_datetime(df["timestamp"], format="%H:%M:%S", errors="coerce")
    t_max = df["_ts"].max()
    cutoff = t_max - pd.Timedelta(minutes=OUTRO_MINUTES)
    return df[df["_ts"] >= cutoff].drop(columns=["_ts"])


def run(df: pd.DataFrame) -> dict:
    """단일 강의 DataFrame → 수업 마무리 요약 컨텍스트 추출.

    Args:
        df: 단일 강의 발화 DataFrame (텍스트 파일 1개분).
            컬럼 기준: timestamp, speaker_id, text_raw

    Returns:
        {"chunk": str} — LLM에 넘길 요약 컨텍스트 텍스트.
    """
    outro_df = _extract_outro(df)
    utterances = [
        Utterance(
            timestamp=str(row.get("timestamp", "")),
            speaker_id=str(row.get("speaker_id", "")),
            text=str(row.get("text_raw", "")),
            seconds_from_start=0,
        )
        for _, row in outro_df.iterrows()
    ]
    payload = build_summary_llm_payload(utterances)
    return {"chunk": payload["context"]}


# ── CLI 진입점 ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    from app.preprocessing.utils import parse_and_split

    parser = argparse.ArgumentParser(description="수업 마무리 요약 컨텍스트 추출")
    parser.add_argument("txt_path", help="원본 강의 텍스트 파일 경로 (단일 강의)")
    parser.add_argument("--output", "-o", default=None, help="결과 JSON 저장 경로 (생략 시 stdout)")
    args = parser.parse_args()

    _df = parse_and_split(args.txt_path)
    result = run(_df)

    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"저장 완료: {args.output}")
    else:
        print(output)
