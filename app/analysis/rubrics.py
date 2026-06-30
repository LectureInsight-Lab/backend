"""항목별 채점 루브릭 로더 (configs/item_rubrics.yaml).

explainer(항목 해설)·narrative(종합 분석)가 LLM 프롬프트에 채점 근거를 주입해
해설을 루브릭에 정합시키는 용도. 파일이 없으면 빈 dict 로 동작(해설은 여전히 생성됨).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

_RUBRIC_PATH = "configs/item_rubrics.yaml"


@lru_cache(maxsize=1)
def load_item_rubrics(path: str | Path = _RUBRIC_PATH) -> dict[int, dict]:
    """{item_id: {criterion, high, low}} 반환. 파일 없으면 {}."""
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    items = data.get("items", {})
    return {int(k): v for k, v in items.items()}


def rubric_line(item_id: int) -> str:
    """단일 항목 루브릭을 한 줄 문자열로 (프롬프트 주입용). 없으면 ''."""
    r = load_item_rubrics().get(item_id)
    if not r:
        return ""
    parts = [f"기준: {r.get('criterion', '')}"]
    if r.get("high"):
        parts.append(f"고득점=({r['high']})")
    if r.get("low"):
        parts.append(f"저득점=({r['low']})")
    if r.get("caveat"):
        parts.append(f"주의=({r['caveat']})")
    return " / ".join(parts)
