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
import math
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
    """anchored_labeled.json 파일을 읽어 키워드를 추출·저장한다 (CLI/배치 진입점).

    실제 추출 로직은 ``build_keywords_from_records`` 가 담당하며, 이 함수는 파일 로드만 한다.
    """
    labeled_path = Path(labeled_path)
    logger.info(f"[Pipeline] 입력: {labeled_path}")
    records = json.loads(labeled_path.read_text(encoding="utf-8"))
    logger.info(f"[Pipeline] 레코드 {len(records)}개 로드")
    return build_keywords_from_records(
        records, output_path=output_path, top_n=top_n, model_name=model_name
    )


def build_keywords_from_records(
    records:            list[dict],
    output_path:        str | Path = KEYWORDS_JSON,
    top_n:              int        = DEFAULT_TOP_N,
    model_name:         str        = KEYBERT_MODEL,
    df_auto_stopwords:  bool       = True,
) -> dict[str, list[tuple[str, float]]]:
    """라벨링 레코드(list[dict]; file·llm_label·text 키)에서 날짜별 top-N 키워드를 추출해
    keywords.json 으로 저장한다.

    파일을 거치지 않고 메모리 레코드를 직접 받으므로 분석 파이프라인이 라벨링 직후 그대로
    호출할 수 있다(항목 7 자동 생성용).

    ⚠️ df_auto_stopwords: '여러 강의를 모았을 때' 모든 강의에 공통 등장하는 단어를 불용어로
    빼는 기능이라 **단일 강의(문서 1개)에서는 모든 명사를 제거**해 키워드가 0개가 된다.
    배치(여러 강의)면 True, 단일 강의 자동 생성이면 반드시 False 로 호출할 것.

    Steps
    -----
    2. NounExtractor (Kiwi) — df_auto_stopwords 옵션 적용
    3. 날짜별 명사 풀 + 텍스트 합산
    4. KeyBERT — 날짜 단위 top-N (모델 한 번만 로드)
    5. keywords.json 저장

    Returns
    -------
    {date: [(word, score), ...]}  score 내림차순
    """
    from keybert import KeyBERT
    from sentence_transformers import SentenceTransformer
    from app.preprocessing.item07_noun_extractor import NounExtractor, normalize_text

    output_path = Path(output_path)

    # ── Step 2: Kiwi 명사 추출 (구간별) ──────────────────────────────────────
    logger.info("[Pipeline] Kiwi 형태소 분석 + 명사 추출 시작")
    noun_ex = NounExtractor(df_auto_stopwords=df_auto_stopwords)
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

    # [Step2 개선] 2026-06-29: IDF 가중치 준비
    # 문제: KeyBERT는 문서 전체를 대표하는 단어(문자열, 인덱스 등 일반어)를 선호하여
    #       특정 날짜에 집중 등장하는 도메인 용어(리플레이스, 트림 등)가 낮은 순위를 받음.
    # 해결: keybert_score × idf(단어) 로 최종 점수를 재산정.
    #       IDF = log(전체날짜수 / 등장날짜수 + 1) + 1
    #       → 많은 날짜에 등장할수록 IDF 낮음 → 최종 점수 낮아짐
    #       → 특정 날짜에만 집중 등장하면 IDF 높음 → 최종 점수 높아짐
    df_counter = getattr(noun_ex, "df_counter", {})
    n_docs     = getattr(noun_ex, "n_docs", 1)

    def idf(word: str) -> float:
        df = df_counter.get(word, 0)
        return math.log(n_docs / (df + 1) + 1)

    results: dict[str, list[tuple[str, float]]] = {}
    for date in sorted(date_nouns):
        nouns = date_nouns[date]
        text  = date_texts.get(date, "").strip()

        if not nouns or not text:
            logger.warning(f"[Pipeline] {date} — 명사 또는 텍스트 없음, 건너뜀")
            results[date] = []
            continue

        # [Step4] TF(날짜 내 등장 빈도) 사전 계산
        tf_map: dict[str, int] = {}
        for noun in nouns:
            tf_map[noun] = text.count(noun)

        def tf_boost(word: str) -> float:
            return math.log(1 + tf_map.get(word, 0))

        # [Step5 개선] 2026-06-29: 가산형(additive) 하이브리드 점수로 전환
        # 문제: 곱셈형 공식(Step4) keybert × idf × tf_boost 에서
        #       트림(rank=242, sim=0.097), 콘캣(rank=184, sim=0.14) 등
        #       KeyBERT 유사도가 낮은 SQL 함수명은 TF가 높아도 최종 점수가 0에 수렴.
        #       BLLB/캐릭터셋/콜레이션은 임베딩 유사도 자체가 0.00 → 완전 제외.
        # 해결: idf × (keybert_sim + GAMMA × log(1+tf)) 가산형으로 변경.
        #       keybert_sim=0 이어도 GAMMA × tf_boost 항이 살아있어 점수 획득 가능.
        #       IDF가 낮은 일반어(문자열, 함수)는 TF가 높아도 전체 점수가 억제됨.
        #       → top_n*5 상한 제거: MIN_KEYBERT_SCORE 필터도 불필요, 전체 후보 스코어링.
        GAMMA = 0.3

        logger.info(f"[Pipeline] {date} — 명사 후보 {len(nouns)}개 → top-{top_n} 추출 (가산형 하이브리드)")
        raw = kb.extract_keywords(
            text,
            candidates=nouns,
            top_n=len(nouns),  # [Step5] 전체 후보 스코어링 (상한 없음)
            use_mmr=False,     # exp05: MMR은 도메인 클러스터에 역효과
        )
        # KeyBERT가 반환하지 않는 후보(유사도=0)는 0.0으로 기본 처리
        raw_score: dict[str, float] = {w: max(s, 0.0) for w, s in raw}
        reranked = sorted(
            [
                (noun, idf(noun) * (raw_score.get(noun, 0.0) + GAMMA * tf_boost(noun)))
                for noun in nouns
            ],
            key=lambda x: x[1],
            reverse=True,
        )[:top_n]
        results[date] = reranked
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
