"""
항목 2-4: 핵심 강조 (keyword_emphasis_rate)

사전 준비:
    python -m app.preprocessing.keyword_pipeline   # keywords.json 생성

파이프라인:
    keywords.json (날짜별 top-N 키워드, 전처리 파이프라인 출력)
        │
        └─ EmphasisChecker : 강조 마커 regex + 반복 탐지 → 키워드별 강조 여부

채점:
    emphasis_rate = 강조 확인된 키워드 수 / 전체 키워드 수
    5점 ≥ 80% / 4점 ≥ 60% / 3점 ≥ 40% / 2점 ≥ 20% / 1점 < 20%

출력:
    {"evidence": list[object] | null, "final_score": int | "N/A"}
    mode=""      → evidence: null
    mode="debug" → evidence: 키워드별 강조 여부 리스트

사용 예:
    from app.analysis.item07_emphasis import score_keyword_emphasis
    result = score_keyword_emphasis("2026-02-02", "data/raw/2026-02-02_kdt-backendj-21th.txt")
"""
from __future__ import annotations

from pathlib import Path

from loguru import logger

from app.preprocessing.item07_keyword_pipeline import KEYWORDS_JSON, load_keywords

ITEM_ID     = 7
ITEM_NAME   = "핵심 강조"
CATEGORY_ID = 2


def run(txt_path) -> dict:
    """원본 STT txt 경로 → 핵심 강조 점수 (항목 7). 파이프라인 진입점.

    날짜는 파일명 stem('{date}_{id}')에서 추출. keywords.json(KeyBERT 전처리 산출)이
    없거나 해당 날짜 키워드가 없으면 N/A 반환(graceful skip).
    """
    date = Path(txt_path).stem.split("_", 1)[0]
    return score_keyword_emphasis(date, txt_path)


def score_keyword_emphasis(
    date:          str,
    txt_path:      str | Path,
    mode:          str = "",
    keywords_path: str | Path = KEYWORDS_JSON,
) -> dict:
    """
    Parameters
    ----------
    date : str
        날짜 문자열 (예: "2026-02-02").
    txt_path : str | Path
        같은 날짜의 raw STT .txt 파일 경로.
    mode : str
        "" (default) → evidence: null / "debug" → evidence: 키워드별 강조 여부
    keywords_path : str | Path
        keyword_pipeline.run_pipeline() 이 생성한 keywords.json 경로.
    """
    from app.analysis.emphasis_checker import EmphasisChecker

    txt_path      = Path(txt_path)
    keywords_path = Path(keywords_path)

    # ── Step 1: 키워드 로드 ──────────────────────────────────────────────────
    if not keywords_path.exists():
        logger.error(f"[item07] keywords.json 없음: {keywords_path}")
        return _build(score="N/A", evidence_items=[], mode=mode)

    all_kw = load_keywords(keywords_path)

    if date not in all_kw:
        logger.warning(f"[item07] keywords.json에 '{date}' 없음")
        return _build(score="N/A", evidence_items=[], mode=mode)

    kw_with_scores = all_kw[date]
    all_keywords   = [w for w, _ in kw_with_scores]

    if not all_keywords:
        logger.warning(f"[item07] {date} 키워드 0개")
        return _build(score="N/A", evidence_items=[], mode=mode)

    logger.info(f"[item07] {date} 키워드 {len(all_keywords)}개 로드: {all_keywords[:6]} …")

    # ── Step 2: 강조 여부 탐지 ──────────────────────────────────────────────
    logger.info(f"[item07] 강조 탐지 시작: {txt_path.name}")
    checker         = EmphasisChecker()
    emphasis_result = checker.check(all_keywords, txt_path)
    summary         = checker.summarize(emphasis_result)

    # ── Step 3: 채점 ────────────────────────────────────────────────────────
    rate      = summary["rate"]
    confirmed = len(summary["emphasized"])
    total     = len(all_keywords)
    rate_pct  = rate * 100

    score = (
        5 if rate_pct >= 80 else
        4 if rate_pct >= 60 else
        3 if rate_pct >= 40 else
        2 if rate_pct >= 20 else
        1
    )

    logger.info(
        f"[item07] {date}: 키워드 {total}개 중 {confirmed}개 강조 확인 "
        f"({rate_pct:.0f}%) → {score}점"
    )

    evidence_items = [
        {"keyword": kw, "emphasized": True}
        for kw in summary["emphasized"]
    ] + [
        {"keyword": kw, "emphasized": False}
        for kw in summary["not_emphasized"]
    ]

    return _build(score=score, evidence_items=evidence_items, mode=mode)


def _build(score: int | str, evidence_items: list[dict], mode: str) -> dict:
    return {
        "evidence":    evidence_items if mode == "debug" else None,
        "final_score": score,
    }
