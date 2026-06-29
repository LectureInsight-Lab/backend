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

from app.analysis import (
    item01_repetition,
    item04_learning_objectives,
    item05_review_linkage,
    item07_emphasis,
    item06_sequence_violation,
    item08_summary,
    item09_concept_definition,
    item10_example_coverage,
    item11_prerequisite,
    item13_example_relevance,
    item14_practice_link,
    item15_error_handling,
    item18_question,
)
from app.preprocessing.utils import PROCESSED_DIR, label_from_csv, parse_and_split

# ── item07 래퍼: (date, txt_path) 시그니처를 (txt_path,) 로 통일 ──────────────
def _run_item07(txt_path: Path) -> dict:
    date = txt_path.stem.split("_")[0]
    return item07_emphasis.score_keyword_emphasis(date, txt_path)


# ── 18개 평가 항목 레지스트리 ─────────────────────────────────────────────────
# (결과 키, callable, 입력 종류, 출력 종류)
#   입력 종류: "kss" | "labeled" | "txt"
#   출력 종류: "score" | "chunk"
_ITEMS: list[tuple[str, object, str, str]] = [
    ("repetition",          item01_repetition.score_repetition,            "txt",     "score"),
    ("question",            question.run,                                  "kss",     "chunk"),
    ("summary",             summary.run,                                   "kss",     "chunk"),
    ("error_handling",      error_handling.run,                            "labeled", "score"),
    ("sequence_violation",  sequence_violation.run,                        "labeled", "score"),
    ("learning_objectives", item04_learning_objectives.run,                "kss",     "score"),
    ("review_linkage",      item05_review_linkage.run,                     "kss",     "score"),
    ("keyword_emphasis",    _run_item07,                                   "txt",     "score"),
    ("concept_definition",  item09_concept_definition.run,                 "labeled", "score"),
    ("example_coverage",    item10_example_coverage.run,                   "labeled", "score"),
    ("prerequisite",        item11_prerequisite.score_prerequisite,        "txt",     "score"),
    ("example_relevance",   item13_example_relevance.run,                  "labeled", "score"),
    ("practice_link",       item14_practice_link.score_practice_link,      "labeled", "score"),
    # TODO: 나머지 5개 항목 추가
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
    # TODO: 나머지 9개 항목 추가
]


def run(txt_path: str | Path, concurrency: int = 10) -> dict:
    """강의 분석 파이프라인 실행.

    Args:
        txt_path:    원본 강의 텍스트 파일 경로 (단일 강의)
        concurrency: Gemini 동시 요청 수

    Returns:
        18개 평가 항목 결과를 병합한 dict.
    """
    txt_path = Path(txt_path)

    # ── 공유 입력 생성 (한 번만) ─────────────────────────────────
    kss_df = parse_and_split(txt_path)
    csv_path = PROCESSED_DIR / f"{txt_path.stem}_kss.csv"
    labeled_df = pd.DataFrame(label_from_csv(csv_path, concurrency=concurrency))

    # ── 전체 항목 실행 + 병합 ────────────────────────────────────
    inputs = {"kss": kss_df, "labeled": labeled_df, "txt": txt_path}
    final_score, chunk = asyncio.run(_run_all(inputs, concurrency))
    inputs = {"kss": kss_df, "labeled": labeled_df}
    details, chunk = asyncio.run(_run_all(inputs, concurrency))

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


async def _run_all(
    inputs: dict[str, pd.DataFrame], concurrency: int
) -> tuple[dict, dict]:
    """레지스트리의 모든 항목을 실행하고 출력 종류별로 두 섹션으로 나눈다.

    Returns:
        (details 섹션, chunk 섹션)
    """
    tasks = [
        _run_item(run_fn, inputs[src], concurrency)
        for _, run_fn, src, _ in _ITEMS
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    details: dict = {}
    chunk: dict = {}
    for (key, _, _, out), res in zip(_ITEMS, results):
        if isinstance(res, Exception):
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