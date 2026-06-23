"""스코어카드 영속화 (경량 파일 스토어).

DB 도입 전까지 분석 결과를 ``data/processed/scorecards/{instructor}/{date}.json`` 에
저장하고 강사별/단일 조회를 제공한다. 리포트 생성·강사 누적 조회·트렌드 산출에 사용.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.analysis.schemas import InstructorScorecard

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STORE_ROOT = PROJECT_ROOT / "data" / "processed" / "scorecards"


def scorecard_id(instructor_id: str, lecture_date: str) -> str:
    return f"{instructor_id}__{lecture_date}"


def _path(instructor_id: str, lecture_date: str, root: Path) -> Path:
    return root / instructor_id / f"{lecture_date}.json"


def save(scorecard: InstructorScorecard, root: Path = STORE_ROOT) -> str:
    """스코어카드 저장 → id 반환."""
    path = _path(scorecard.instructor_id, scorecard.lecture_date, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(scorecard.model_dump_json(indent=2), encoding="utf-8")
    return scorecard_id(scorecard.instructor_id, scorecard.lecture_date)


def get(card_id: str, root: Path = STORE_ROOT) -> InstructorScorecard | None:
    """id (instructor__date) 로 단일 조회."""
    if "__" not in card_id:
        return None
    instructor_id, lecture_date = card_id.split("__", 1)
    path = _path(instructor_id, lecture_date, root)
    if not path.exists():
        return None
    return InstructorScorecard.model_validate_json(path.read_text(encoding="utf-8"))


def list_by_instructor(instructor_id: str, root: Path = STORE_ROOT) -> list[InstructorScorecard]:
    """강사별 누적 스코어카드 (강의일 오름차순)."""
    folder = root / instructor_id
    if not folder.exists():
        return []
    cards = [
        InstructorScorecard.model_validate_json(p.read_text(encoding="utf-8"))
        for p in folder.glob("*.json")
    ]
    return sorted(cards, key=lambda c: c.lecture_date)
