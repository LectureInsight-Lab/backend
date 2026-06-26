"""
전처리 통합 파이프라인

anchored_labeled.json
  └─ Kiwi 형태소 분석 + 명사 추출  (NounExtractor, df_auto_stopwords=True)
  └─ KeyBERT 날짜별 top-N 추출     (jhgan/ko-sroberta-multitask, use_mmr=False)
  └─ keywords.json 저장

구간(개념/예시/실습)별이 아닌 날짜 단위로 묶어서 KeyBERT를 한 번씩만 실행한다.
모델은 전체 파이프라인에서 한 번만 로드한다.

실행:
    python -m app.preprocessing.keyword_pipeline
    python -m app.preprocessing.keyword_pipeline --top-n 12
    python -m app.preprocessing.keyword_pipeline --input data/processed/anchored_labeled.json --output data/processed/keywords.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from loguru import logger

# ── 경로 / 상수 ──────────────────────────────────────────────────────────────
_BASE          = Path(__file__).parent.parent.parent
LABELED_JSON   = _BASE / "data" / "processed" / "anchored_labeled.json"
KEYWORDS_JSON  = _BASE / "data" / "processed" / "keywords.json"

KEYBERT_MODEL  = "jhgan/ko-sroberta-multitask"
DEFAULT_TOP_N  = 10
LABEL_ORDER    = ["개념", "예시", "실습"]


# ── 내부 헬퍼 ────────────────────────────────────────────────────────────────

def _group_by_date(records: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        fname = rec.get("file", "")
        date  = fname.split("_")[0] if fname else rec.get("date", "")
        if date:
            grouped[date].append(rec)
    return dict(grouped)


# ── 핵심 함수 ────────────────────────────────────────────────────────────────

def run_pipeline(
    labeled_path: str | Path = LABELED_JSON,
    output_path:  str | Path = KEYWORDS_JSON,
    top_n:        int        = DEFAULT_TOP_N,
    model_name:   str        = KEYBERT_MODEL,
) -> dict[str, list[tuple[str, float]]]:
    """
    전체 파이프라인 실행 후 keywords.json 에 저장한다.

    Steps
    -----
    1. anchored_labeled.json 로드
    2. NounExtractor (Kiwi) — 전체 레코드, df_auto_stopwords=True
    3. 날짜별 명사 풀 + 텍스트 합산
    4. KeyBERT — 날짜 단위 top-N (모델 한 번만 로드)
    5. keywords.json 저장

    Returns
    -------
    {date: [(word, score), ...]}  score 내림차순
    """
    from keybert import KeyBERT
    from sentence_transformers import SentenceTransformer
    from app.preprocessing.noun_extractor import NounExtractor, normalize_text

    labeled_path = Path(labeled_path)
    output_path  = Path(output_path)

    # ── Step 1: 레코드 로드 ───────────────────────────────────────────────────
    logger.info(f"[Pipeline] 입력: {labeled_path}")
    records = json.loads(labeled_path.read_text(encoding="utf-8"))
    logger.info(f"[Pipeline] 레코드 {len(records)}개 로드")

    # ── Step 2: Kiwi 명사 추출 (구간별) ──────────────────────────────────────
    logger.info("[Pipeline] Kiwi 형태소 분석 + 명사 추출 시작")
    noun_ex = NounExtractor(df_auto_stopwords=True)
    section_nouns = noun_ex.extract(records)   # {(date, label): [nouns]}

    # ── Step 3: 날짜별 명사 풀 + 텍스트 합산 ─────────────────────────────────
    date_nouns: dict[str, list[str]] = defaultdict(list)
    for (date, _), nouns in section_nouns.items():
        date_nouns[date].extend(nouns)

    # 날짜별 중복 제거 (등장 순서 유지)
    for date in date_nouns:
        seen: dict[str, None] = {}
        for w in date_nouns[date]:
            seen[w] = None
        date_nouns[date] = list(seen)

    date_texts: dict[str, str] = defaultdict(str)
    for rec in records:
        fname  = rec.get("file", "")
        date   = fname.split("_")[0] if fname else rec.get("date", "")
        label  = rec.get("llm_label", "기타")
        if date and label in LABEL_ORDER:
            date_texts[date] += " " + normalize_text(rec.get("text", ""))

    # ── Step 4: KeyBERT (모델 한 번 로드) ────────────────────────────────────
    logger.info(f"[Pipeline] 모델 로딩: {model_name}")
    kb = KeyBERT(model=SentenceTransformer(model_name))

    results: dict[str, list[tuple[str, float]]] = {}
    for date in sorted(date_nouns):
        nouns = date_nouns[date]
        text  = date_texts.get(date, "").strip()

        if not nouns or not text:
            logger.warning(f"[Pipeline] {date} — 명사 또는 텍스트 없음, 건너뜀")
            results[date] = []
            continue

        logger.info(f"[Pipeline] {date} — 명사 후보 {len(nouns)}개 → top-{top_n} 추출")
        raw = kb.extract_keywords(
            text,
            candidates=nouns,
            top_n=top_n,
            use_mmr=False,   # exp05: MMR은 도메인 클러스터에 역효과
        )
        results[date] = sorted(raw, key=lambda x: x[1], reverse=True)
        logger.debug(f"  → {[w for w, _ in results[date]]}")

    # ── Step 5: JSON 저장 ─────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        date: [[w, round(s, 6)] for w, s in kws]
        for date, kws in results.items()
    }
    output_path.write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"[Pipeline] 저장 완료: {output_path} ({len(results)}개 날짜)")

    return results


# ── 로드 헬퍼 ────────────────────────────────────────────────────────────────

def load_keywords(
    path: str | Path = KEYWORDS_JSON,
) -> dict[str, list[tuple[str, float]]]:
    """
    저장된 keywords.json 을 로드한다.

    Returns
    -------
    {date: [(word, score), ...]}
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {date: [(w, float(s)) for w, s in kws] for date, kws in data.items()}


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="키워드 추출 전처리 파이프라인")
    parser.add_argument("--input",  default=str(LABELED_JSON),  help="anchored_labeled.json 경로")
    parser.add_argument("--output", default=str(KEYWORDS_JSON), help="keywords.json 출력 경로")
    parser.add_argument("--top-n",  type=int, default=DEFAULT_TOP_N, help="날짜별 추출 키워드 수 (기본: 10)")
    parser.add_argument("--model",  default=KEYBERT_MODEL,      help="sentence-transformers 모델")
    args = parser.parse_args()

    run_pipeline(args.input, args.output, args.top_n, args.model)
