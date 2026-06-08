"""체크리스트 메타 로더.

`configs/checklist.yaml` 을 읽어 18개 항목의 메타데이터를 제공한다.
각 항목은 ChecklistItem 으로 표현되며, item_type 에 따라 ensemble 전략이 갈린다.
"""
from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ItemType(str, Enum):
    """Paper #4: LLM이 잘하는 영역 vs 보조 신호 필요한 영역 구분."""

    DISCRETE = "discrete"                 # 존재 여부 판단 → LLM 직접 사용
    HIGH_INFERENCE = "high_inference"     # 질적 판단 → LLM + BoW 앙상블


class ContextStrategy(str, Enum):
    """LLM 프롬프트에 어느 구간의 발화를 컨텍스트로 줄지 결정."""

    INTRO = "intro"               # 시작 후 20분
    MIDDLE = "middle"             # 중간 구간
    OUTRO = "outro"               # 종료 전 15분
    FULL_SAMPLE = "full_sample"   # 전체에서 RAG로 샘플링
    KEYWORD = "keyword"           # 키워드 기반 발화 선택


class ChecklistItem(BaseModel):
    """단일 체크리스트 항목 메타."""

    id: int
    name: str
    category: str                 # structure / concept / practice / language / interaction
    item_type: ItemType
    context_strategy: ContextStrategy
    prompt: str                   # 프롬프트 YAML 경로


class Checklist(BaseModel):
    """체크리스트 전체 (18 항목 + 카테고리 가중치 + 구간 분리 기준)."""

    items: list[ChecklistItem]
    category_weights: dict[str, float]
    segments: dict[str, int]      # {"intro_minutes": 20, "outro_minutes": 15}

    def by_id(self, item_id: int) -> ChecklistItem:
        return next(i for i in self.items if i.id == item_id)

    def by_category(self, category: str) -> list[ChecklistItem]:
        return [i for i in self.items if i.category == category]


def load_checklist(path: str | Path = "configs/checklist.yaml") -> Checklist:
    """YAML → Checklist 모델."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Checklist(**data)
