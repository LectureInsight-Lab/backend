"""강의 분석 전체 파이프라인 (18개 평가 항목 통합).

흐름 (입력: 단일 강의 txt 파일):
    txt 파일
      → parse_and_split()  : KSS 문장 분리 → <stem>_kss.csv   (공유 입력 "kss")
      → label_from_csv()   : 개념/예시/실습 라벨링 → <stem>_labeled.json (공유 입력 "labeled")
      → 각 평가 항목 run() 실행 (kss / labeled 중 하나를 입력으로 사용)
      → 18개 결과를 하나의 dict로 병합해 반환 (파일 저장 없음)

평가 항목 추가법:
    아래 _ITEMS 리스트에 (결과키, 모듈, 입력종류, 출력종류) 한 줄 추가.
    - 입력종류 "kss"     : raw KSS 문장 DataFrame
    - 입력종류 "labeled" : 라벨링된 청크 DataFrame
    - 출력종류 "score"   : 최종 점수를 내는 항목 → 결과의 "final_score" 섹션
    - 출력종류 "chunk"   : LLM 평가용 chunk만 내는 항목 → 결과의 "chunk" 섹션
    각 모듈은 run(df) 또는 run(df, concurrency=...) 시그니처면 되고,
    sync / async 여부는 자동 감지된다.

Usage:
    from app.analysis.pipeline import run

    result = run("data/raw/2026-02-02_kdt-backendj-21th.txt")
"""

from __future__ import annotations

import asyncio
import inspect
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from app.analysis import (
    item04_learning_objectives,
    item05_review_linkage,
    item06_sequence_violation,
    item08_summary,
    item09_concept_definition,
    item10_example_coverage,
    item13_example_relevance,
    item14_practice_link,
    item15_error_handling,
    item18_question,
)
from app.core.jobs import NULL_SINK, ProgressSink
from app.preprocessing import (
    item02_completeness,
    item03_consistency,
    item12_pace,
    item16_comprehension_check,
    item17_engagement,
    sentencizer,
)
from app.preprocessing.utils import (
    PROCESSED_DIR,
    build_utterances,
    label_from_csv,
    parse_and_split,
)

# ── 18개 평가 항목 레지스트리 ─────────────────────────────────────────────────
# (결과 키, 모듈, 입력 종류, 출력 종류)
#   입력 종류: "kss" | "labeled" | "sentences"(Kiwi 문장화, 항목 2·3)
#   출력 종류: "score" | "chunk"
#   항목 4·5·9·10·13: app.analysis.itemNN_* 모듈이 내부에서 preprocessing chunk emitter를
#   호출해 LLM 채점까지 수행 → "score" 타입.
_ITEMS: list[tuple[str, object, str, str]] = [
    # item04
    ("learning_objectives", item04_learning_objectives, "kss",     "score"),
    # item05
    ("review_linkage",      item05_review_linkage,      "kss",     "score"),
    # item06 — analysis 모듈이 내부에서 preprocessing 호출
    ("sequence_violation",  item06_sequence_violation,  "labeled", "score"),
    # item08 — preprocessing → Gemini 채점: 마무리 요약 충실도
    ("summary",             item08_summary,             "kss",     "score"),
    # item09
    ("concept_definition",  item09_concept_definition,  "labeled", "score"),
    # item10
    ("example_coverage",    item10_example_coverage,    "labeled", "score"),
    # item13
    ("example_relevance",   item13_example_relevance,   "labeled", "score"),
    # item15 — analysis 모듈이 내부에서 preprocessing 호출
    ("error_handling",      item15_error_handling,      "labeled", "score"),
    # item18 — preprocessing → Gemini 채점: 질문 응답 충분성
    ("question",            item18_question,            "kss",     "score"),
    # item14 — analysis 모듈, labeled 청크로 이론↔실습 LLM 비교 채점
    ("practice_link",       item14_practice_link,       "labeled", "score"),
    # item12 — preprocessing, 발화 속도(분당 음절) 규칙 채점
    ("pace",                item12_pace,                "kss",     "score"),
    # item16 — preprocessing, 이해 확인 질문 빈도/타이밍 규칙 채점
    ("comprehension",       item16_comprehension_check, "kss",     "score"),
    # item17 — preprocessing, 참여 유도 빈도/대기/피드백 규칙 채점
    ("engagement",          item17_engagement,          "kss",       "score"),
    # item02 — Kiwi 문장화 → 완결 문장 비율(EF 종결) 규칙 채점
    ("completeness",        item02_completeness,        "sentences", "score"),
    # item03 — Kiwi 문장화 → 말투 일관성(존/반/중립) 규칙 채점
    ("consistency",         item03_consistency,         "sentences", "score"),
    # TODO: 나머지 항목(01·07·11) 추가
]


def run(
    txt_path: str | Path,
    concurrency: int = 10,
    progress: ProgressSink | None = None,
) -> dict:
    """강의 분석 파이프라인 실행.

    Args:
        txt_path:    원본 강의 텍스트 파일 경로 (단일 강의)
        concurrency: Gemini 동시 요청 수
        progress:    진행률 리포터 (없으면 무시). 항목 단위 완료를 보고한다.

    Returns:
        18개 평가 항목 결과를 병합한 dict.
    """
    progress = progress or NULL_SINK
    txt_path = Path(txt_path)
    logger.info(f"[pipeline] ▶ 분석 시작 — 입력: {txt_path.name} (concurrency={concurrency})")

    # ── 공유 입력 생성 (한 번만) ─────────────────────────────────
    kss_df = parse_and_split(txt_path)
    logger.info(f"[pipeline] 1/3 문장 분리(KSS) 완료 — {len(kss_df)} 문장")
    progress.stage(f"문장 분리(KSS) 완료 — {len(kss_df)} 문장", 8)
    csv_path = PROCESSED_DIR / f"{txt_path.stem}_kss.csv"
    labeled_df = pd.DataFrame(label_from_csv(csv_path, concurrency=concurrency))
    logger.info(f"[pipeline] 2/3 개념/예시/실습 라벨링 완료 — {len(labeled_df)} 청크")
    progress.stage(f"개념/예시/실습 라벨링 완료 — {len(labeled_df)} 청크", 16)

    # 항목 2·3 공유 입력: Kiwi 문장화(분리+EF/EC 완결성 판정) — 한 번만
    sentences = sentencizer.build_sentences(build_utterances(kss_df))
    logger.info(f"[pipeline] Kiwi 문장화 완료 — {len(sentences)} 문장 (항목 2·3용)")

    # ── 전체 항목 실행 + 병합 ────────────────────────────────────
    inputs = {"kss": kss_df, "labeled": labeled_df, "sentences": sentences}
    logger.info(f"[pipeline] 3/3 평가 항목 {len(_ITEMS)}개 병렬 실행…")
    progress.items(len(_ITEMS))
    progress.stage(f"평가 항목 {len(_ITEMS)}개 분석 중…", 16)
    details, chunk = asyncio.run(_run_all(inputs, concurrency, progress))
    logger.success(
        "[pipeline] ✓ 분석 완료 — scores: "
        + ", ".join(
            f"{k}={v.get('final_score') if isinstance(v, dict) else None}"
            for k, v in details.items()
        )
    )

    return {
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "source": txt_path.name,
        "scores": {
            key: (val.get("final_score") if isinstance(val, dict) else None)
            for key, val in details.items()
        },
        "details": details,
        "chunk": chunk,
    }


# ── 라우트 진입점: raw_text → InstructorScorecard ────────────────────────────
# pipeline.run() 결과 dict 의 항목 키 → 체크리스트 항목 id 매핑.
# (현재 동작 검증된 9개 항목. 나머지 9개는 프롬프트 라이브러리 완성 후 추가 — TODO)
_KEY_TO_ID: dict[str, int] = {
    "learning_objectives": 4,
    "review_linkage": 5,
    "sequence_violation": 6,
    "summary": 8,
    "concept_definition": 9,
    "example_coverage": 10,
    "example_relevance": 13,
    "error_handling": 15,
    "question": 18,
    "practice_link": 14,
    "pace": 12,
    "comprehension": 16,
    "engagement": 17,
    "completeness": 2,
    "consistency": 3,
}


def _detail_to_item_score(key: str, detail: object, checklist):
    """run() 의 항목 결과 dict → ItemScore. 점수 없음/비수치('N/A', error)면 None."""
    from app.analysis.schemas import ItemScore

    item_id = _KEY_TO_ID.get(key)
    if item_id is None or not isinstance(detail, dict):
        return None
    try:
        score = float(detail.get("final_score"))
    except (TypeError, ValueError):
        logger.warning(
            f"[pipeline]   ⚠ [{key}] final_score 없음/비수치({detail.get('final_score')!r}) "
            "→ 스코어카드 제외"
        )
        return None

    score = round(max(1.0, min(5.0, score)), 4)
    meta = checklist.by_id(item_id)

    # basis: 항목 표 '해설' 둘째 줄(근거)의 원천. debug 모드 evidence 인용이 있으면 그걸,
    # 없으면 모듈의 reason(규칙=지표 / LLM=CoT 판정 근거)을 사용.
    detail_reason = str(detail.get("reason") or "")
    ev = detail.get("evidence")
    basis = (
        " / ".join(str(e) for e in ev if e)
        if isinstance(ev, list) and ev
        else detail_reason
    )
    # item14 등은 멀티청크 CoT 라 근거가 매우 길다 → 두 단계로 길이 제한.
    #   raw_basis(explainer 입력): 240자까지 — 한 줄 근거 생성용 맥락
    #   display(근거 줄 폴백): 90자까지 — explainer 가 항목을 누락해도 표가 안 늘어남
    raw_basis = basis[:240].rstrip() + "…" if len(basis) > 240 else basis
    display = basis[:90].rstrip() + "…" if len(basis) > 90 else basis
    return ItemScore(
        item_id=item_id,
        name=meta.name,
        category=meta.category,
        item_type=meta.item_type.value,
        final_score=score,
        llm_score=score,          # 9항목 경로는 LLM 직접 채점 (BoW 앙상블 분리 없음)
        bow_score=0.0,
        final_confidence=1.0,
        evidence=display,              # 근거 줄 폴백 (explainer 가 grounds 로 덮어씀)
        strengths="",
        improvements=raw_basis,        # explainer 입력(raw_basis)
        needs_human_review=False,
    )


async def analyze_raw_text(
    raw_text: str,
    lecture_date: str,
    instructor_id: str,
    concurrency: int = 10,
    progress: ProgressSink | None = None,
):
    """STT 원문(raw_text) → ``InstructorScorecard``  (분석 라우트 진입점).

    동작이 검증된 9개 항목 파이프라인(``run``)을 재사용해 채점하고, 결과 dict 를
    ``InstructorScorecard`` 로 변환한다. ``run`` 은 txt 파일 경로를 입력으로 받고
    내부에서 ``asyncio.run`` 을 호출하므로:
      1) raw_text 를 임시 txt 로 기록하고
      2) 실행 중인 이벤트 루프와 충돌하지 않도록 별도 스레드에서 돌린다.

    NOTE: 18개 항목 스코어카드 경로(analyzer→ensemble)는 프롬프트 라이브러리
    (system.yaml + 항목 YAML 18개) 미완성으로 보류. 완성 시 이 함수만 교체하면 된다.
    """
    import os
    import tempfile

    from app.analysis import explainer, scorer
    from app.core.checklist import load_checklist

    progress = progress or NULL_SINK
    logger.info(
        f"[pipeline] ▶ 실분석 시작 — instructor={instructor_id} date={lecture_date} "
        f"({len(raw_text)}자)"
    )
    progress.stage("전처리 준비 중…", 3)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    # parse_and_split() 은 파일명 stem 의 첫 "_" 앞을 '날짜'로 떼어내므로
    # (date, *rest = stem.split("_", 1)), 임시파일 stem 을 반드시 lecture_date 로 시작시킨다.
    # 안 그러면 date 컬럼이 'api' 같은 값이 되어 pd.to_datetime 에서 실패한다.
    safe_date = lecture_date.replace("/", "-").replace(" ", "")
    fd, tmp_name = tempfile.mkstemp(suffix=".txt", prefix=f"{safe_date}_", text=True)
    tmp_path = Path(tmp_name)
    csv_path = PROCESSED_DIR / f"{tmp_path.stem}_kss.csv"
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(raw_text)
        result = await asyncio.to_thread(run, tmp_path, concurrency, progress)
    finally:
        tmp_path.unlink(missing_ok=True)
        csv_path.unlink(missing_ok=True)

    progress.stage("스코어카드 생성 중…", 96)
    checklist = load_checklist()
    item_scores = [
        s
        for key, detail in result["details"].items()
        if (s := _detail_to_item_score(key, detail, checklist)) is not None
    ]
    logger.info(f"[pipeline] 스코어카드 변환 — 항목 {len(item_scores)}개 반영")

    card = scorer.build_scorecard(instructor_id, lecture_date, item_scores, checklist)

    # 표현 레이어: 규칙엔진 지표 → 자연어 강점/개선점 (Gemini 1회, 실패해도 원본 유지)
    progress.stage("자연어 해설 생성 중…", 97)
    await explainer.attach_explanations(card)

    logger.success(
        f"[pipeline] ✓ 실분석 완료 — overall={card.overall_score} "
        f"(항목 {len(card.item_scores)}개)"
    )
    return card


async def _run_all(
    inputs: dict[str, pd.DataFrame], concurrency: int, progress: ProgressSink = NULL_SINK
) -> tuple[dict, dict]:
    """레지스트리의 모든 항목을 실행하고 출력 종류별로 두 섹션으로 나눈다.

    Returns:
        (details 섹션, chunk 섹션)
    """
    total = len(_ITEMS)
    completed = 0

    async def _logged(key: str, run_fn, df: pd.DataFrame):
        nonlocal completed
        logger.info(f"[pipeline]   ▷ [{key}] 채점 시작 (입력 {len(df)} 행)")
        res = await _run_item(run_fn, df, concurrency)
        completed += 1
        score = res.get("final_score") if isinstance(res, dict) else None
        logger.info(f"[pipeline]   ◁ [{key}] 채점 완료 — final_score={score}")
        progress.item_done(key, score, completed, total)
        return res

    tasks = [
        _logged(key, module.run, inputs[src])
        for key, module, src, _ in _ITEMS
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    details: dict = {}
    chunk: dict = {}
    for (key, _, _, out), res in zip(_ITEMS, results):
        if isinstance(res, Exception):
            logger.error(f"[pipeline]   ✗ [{key}] 실패 — {type(res).__name__}: {res}")
            res = {"error": f"{type(res).__name__}: {res}"}
        (details if out == "score" else chunk)[key] = res
    return details, chunk


async def _run_item(run_fn, df: pd.DataFrame, concurrency: int):
    """모듈 run()을 sync/async·concurrency 유무에 맞춰 호출."""
    kwargs = {}
    if "concurrency" in inspect.signature(run_fn).parameters:
        kwargs["concurrency"] = concurrency

    if inspect.iscoroutinefunction(run_fn):
        return await run_fn(df, **kwargs)
    return await asyncio.to_thread(run_fn, df, **kwargs)


# ── CLI 진입점 ────────────────────────────────────────────────────────────────
# 반드시 프로젝트 루트(Lecturesight-Lab)에서 실행:
#     python -m app.analysis.pipeline data/raw/2026-02-02_kdt-backendj-21th.txt

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="강의 분석 통합 파이프라인 (18개 항목)")
    parser.add_argument("txt_path", help="원본 강의 텍스트 파일 경로 (단일 강의)")
    parser.add_argument("--concurrency", "-c", type=int, default=10, help="Gemini 동시 요청 수 (기본 10)")
    args = parser.parse_args()

    result = run(args.txt_path, concurrency=args.concurrency)
    print("scores:")
    for key, score in result["scores"].items():
        print(f"  {key}: {score}")
    if result["chunk"]:
        print(f"chunk: {list(result['chunk'].keys())}")