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
    labeled_path:    str | Path = LABELED_JSON,
    output_path:     str | Path = KEYWORDS_JSON,
    top_n:           int        = DEFAULT_TOP_N,
    model_name:      str        = KEYBERT_MODEL,
    fragment_dedup:  bool       = True,
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
    from app.preprocessing.item07_noun_extractor import NounExtractor, normalize_text, GOLD_WHITELIST

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

    date_texts:      dict[str, str] = defaultdict(str)  # KeyBERT 문서 (labeled 구간만)
    date_texts_full: dict[str, str] = defaultdict(str)  # TF 집계용 (기타 포함 전체)
    for rec in records:
        fname  = rec.get("file", "")
        date   = fname.split("_")[0] if fname else rec.get("date", "")
        label  = rec.get("llm_label", "기타")
        if date:
            normalized = normalize_text(rec.get("text", ""))
            date_texts_full[date] += " " + normalized  # [Step26] 전체 섹션 TF
            if label in LABEL_ORDER:
                date_texts[date] += " " + normalized   # KeyBERT는 labeled만 유지

    # [Step26] GOLD_WHITELIST 중 기타 섹션에만 등장하는 단어를 후보 풀에 주입
    for date in list(date_nouns.keys()):
        full_text = date_texts_full.get(date, "")
        existing  = set(date_nouns[date])
        for word in GOLD_WHITELIST:
            if word not in existing and word in full_text:
                date_nouns[date].append(word)

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

    # [Step21 개선] 2026-06-30: GOLD_WHITELIST 최소 IDF 하한
    # 문제: 업데이트/인서트/딜리트/락 등 DML·TCL 명령어는 전체 날짜에 걸쳐 등장하므로
    #       IDF≈0.66으로 억제됨. 02-13 강의에서 TF=75(업데이트) 등 높은 빈도임에도
    #       노이즈(ACID 속성어 등 IDF≈0.66×high_TF~1.72) 아래로 밀려 top-10 미진입.
    # 해결: GOLD_WHITELIST 단어는 IDF 하한을 0.8로 보장.
    #       IDF=0.8 → 업데이트(TF=75) score=0.8×(sim+2.17)≈1.97 > 노이즈 임계(1.72) ✅
    #       비-GOLD 일반어의 IDF는 그대로 유지 → 도메인 특이성 억제 효과 유지.
    MIN_GOLD_IDF = 1.3

    def idf(word: str) -> float:
        df = df_counter.get(word, 0)
        raw = math.log(n_docs / (df + 1) + 1)
        if word in GOLD_WHITELIST:
            return max(raw, MIN_GOLD_IDF)
        return raw

    results: dict[str, list[tuple[str, float]]] = {}
    for date in sorted(date_nouns):
        nouns = date_nouns[date]
        text  = date_texts.get(date, "").strip()

        if not nouns or not text:
            logger.warning(f"[Pipeline] {date} — 명사 또는 텍스트 없음, 건너뜀")
            results[date] = []
            continue

        # [Step4] TF(날짜 내 등장 빈도) 사전 계산
        # [Step26] labeled TF=0 이면 기타 포함 전체 TF를 fallback으로 사용.
        # labeled에 등장하는 단어는 labeled TF 유지(기타 노이즈 차단),
        # labeled에 없는 단어(기타 섹션 전용 골드 키워드)는 기타 TF로 후보 진입 가능.
        tf_text_full = date_texts_full.get(date, "")
        tf_map: dict[str, int] = {}
        for noun in nouns:
            tf_lbl = text.count(noun)
            tf_map[noun] = tf_lbl if tf_lbl > 0 else tf_text_full.count(noun)

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
        # [Step9 개선] 2026-06-29: unigram/n-gram GAMMA 분리 적용
        # 문제: GAMMA를 일괄 상향(0.3→0.5)하면 n-gram 후보(문자열 함수, 공백 제거 등)가
        #       df=0 → idf=최대(2.77)이면서 TF도 높아 2중 부스트 → 임계값이 더 빠르게 상승.
        # 해결: unigram TF 가중 GAMMA=0.5, n-gram TF 가중 GAMMA_NG=0.1 로 분리.
        #       n-gram은 이미 IDF 최대 혜택을 받으므로 TF 부스트를 억제.
        #       → 콘캣(unigram, TF=6): 1.386×(0.14+0.5×log(7))=1.544 예상.
        #       → 문자열 함수(bigram, TF=15): idf×(sim+0.1×tf) = 억제.
        GAMMA    = 0.5   # unigram TF 가중
        GAMMA_NG = 0.1   # n-gram TF 가중 (이미 IDF 최대값 → 이중 부스트 방지)

        # [Step8 개선] 2026-06-29: NounExtractor가 n-gram을 포함하여 반환하므로
        # 후보 수가 unigram 단독 대비 2~4배 증가. top_n=len(nouns) 로 전체 스코어링.
        # n-gram의 IDF는 df_counter에 미등록(df=0)이므로 최대 IDF(≈2.77) 자동 부여.
        # n-gram TF: text.count(bigram) 은 정확히 해당 문구 등장 횟수를 반환.
        logger.info(f"[Pipeline] {date} — 명사 후보 {len(nouns)}개 → top-{top_n} 추출 (n-gram 포함)")
        raw = kb.extract_keywords(
            text,
            candidates=nouns,
            top_n=len(nouns),  # [Step5] 전체 후보 스코어링 (상한 없음)
            use_mmr=False,     # exp05: MMR은 도메인 클러스터에 역효과
        )
        # KeyBERT가 반환하지 않는 후보(유사도=0)는 0.0으로 기본 처리
        raw_score: dict[str, float] = {w: max(s, 0.0) for w, s in raw}

        # [Step26] 프리픽스 fragment TF 보정 + dedup
        gw_in_pool = GOLD_WHITELIST & set(nouns)

        if fragment_dedup:
            # raw all_scored → raw top-N 기준으로 가드 판단
            all_scored_raw = sorted(
                [
                    (
                        noun,
                        idf(noun) * (
                            raw_score.get(noun, 0.0)
                            + (GAMMA_NG if " " in noun else GAMMA) * tf_boost(noun)
                        ),
                    )
                    for noun in nouns
                ],
                key=lambda x: x[1],
                reverse=True,
            )
            raw_top_n_set = {w for w, _ in all_scored_raw[:top_n]}

            # fragment TF 차감: GW compound가 raw top-N에 있을 때만
            for noun in list(tf_map):
                if noun in GOLD_WHITELIST or tf_map[noun] == 0:
                    continue
                for gw_word in GOLD_WHITELIST:
                    if gw_word.startswith(noun) and len(gw_word) > len(noun) and gw_word in raw_top_n_set:
                        tf_map[noun] = max(0, tf_map[noun] - text.count(gw_word))
                        break

        all_scored = sorted(
            [
                (
                    noun,
                    idf(noun) * (
                        raw_score.get(noun, 0.0)
                        + (GAMMA_NG if " " in noun else GAMMA) * tf_boost(noun)
                    ),
                )
                for noun in nouns
            ],
            key=lambda x: x[1],
            reverse=True,
        )

        # dedup: TF 보정 후 adj top-N 집합 기준 (보정으로 복합어가 진입한 경우 포착)
        adj_top_n_set = {w for w, _ in all_scored[:top_n]}

        reranked = []
        for word, score in all_scored:
            if len(reranked) >= top_n:
                break
            if fragment_dedup and word not in GOLD_WHITELIST:
                is_frag = any(
                    gw.startswith(word) and len(gw) > len(word) and gw in adj_top_n_set
                    for gw in gw_in_pool
                )
                if is_frag:
                    logger.debug(f"  [dedup] {date}: '{word}' → fragment of GW word in top-N, skipping")
                    continue
            reranked.append((word, score))

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
