"""항목 프롬프트 YAML 로더.

``app/analysis/prompts/items/<item>.yaml`` 의 system / user_template / criteria 등을
dict 로 로드한다. ``item14_practice_link`` 가 사용하며, lru_cache 로 재로드를 막는다.

NOTE: 옛 v2 프롬프트 조립기(build_messages / few_shot / rag_context)는 제거되었다.
LLM 채점 프롬프트는 각 항목 모듈(또는 rubric_llm)이 직접 구성한다.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=32)
def load_item_prompt(prompt_path: str) -> dict:
    """단일 항목 프롬프트 YAML 로드 (프로젝트 루트 기준 상대경로)."""
    path = Path(prompt_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
