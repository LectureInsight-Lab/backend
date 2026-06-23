"""analysis 단계 전체 통합 진입점.

preprocessor 산출물(LectureDocument)을 받아 분석 5단계를 순서대로 실행하고
InstructorScorecard 를 반환한다. API 라우트/대시보드가 이 진입점을 호출한다.

흐름:
    LectureDocument
      → behavior_tagger.tag        (2-A: BoW)
      → embedder.build_index       (2-B: 키워드 인덱스)
      → analyzer.analyze_lecture   (3: Gemini × 18 병렬)
      → ensemble.ensemble_all      (4: 앙상블)
      → scorer.build_scorecard     (5: 카테고리/종합)
"""
from __future__ import annotations

from app.analysis import analyzer, behavior_tagger, embedder, ensemble, scorer
from app.analysis.schemas import InstructorScorecard, LectureDocument
from app.core.checklist import Checklist, load_checklist
from app.preprocessing import preprocessor


async def analyze_document(
    document: LectureDocument,
    checklist: Checklist | None = None,
) -> InstructorScorecard:
    """전처리된 LectureDocument → InstructorScorecard (분석 2~5단계)."""
    checklist = checklist or load_checklist()

    behavior = behavior_tagger.tag(document)
    index = embedder.build_index(document)
    raws = await analyzer.analyze_lecture(document, index, behavior, checklist)
    item_scores = ensemble.ensemble_all(raws, behavior, checklist)
    return scorer.build_scorecard(
        instructor_id=document.instructor_id,
        lecture_date=document.lecture_date,
        item_scores=item_scores,
        checklist=checklist,
    )


async def analyze_raw_text(
    raw_text: str,
    lecture_date: str,
    instructor_id: str,
    checklist: Checklist | None = None,
) -> InstructorScorecard:
    """STT 원문 → 전처리(1단계) → 분석(2~5단계) → InstructorScorecard."""
    document = preprocessor.build_document(
        raw_text, lecture_date=lecture_date, instructor_id=instructor_id, with_sentences=True
    )
    return await analyze_document(document, checklist)
