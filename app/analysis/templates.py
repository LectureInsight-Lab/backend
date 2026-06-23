"""프롬프트 조립기 (Paper #2, #6).

항목별 평가 기준(루브릭) + Few-shot 예시 + RAG 컨텍스트 + BoW 카운트를 합쳐
LLM 에 보낼 메시지를 만든다.

구조:
    [system] 공통 루브릭 (prompts/system.yaml)
    [user]   항목 user_template 에 아래 변수를 치환:
             few_shot_examples / rag_context / positive_count / negative_count /
             bow_score / top_k / filler_word_ratio / avg_line_gap_seconds /
             completeness_rate / consistency_ratio
             — 템플릿이 참조하지 않는 변수는 무시되고, 참조하지만 값이 없는 변수는 빈 문자열.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from app.analysis.schemas import IndexedChunk, ItemBoW, LectureDocument
from app.core.checklist import ChecklistItem

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SYSTEM_PROMPT = PROJECT_ROOT / "app" / "analysis" / "prompts" / "system.yaml"
FEW_SHOT_CONFIG = PROJECT_ROOT / "configs" / "few_shot_examples.yaml"


class _SafeDict(dict):
    """str.format_map 용 — 없는 키는 빈 문자열로 대체(KeyError 방지)."""

    def __missing__(self, key: str) -> str:
        return ""


@lru_cache(maxsize=1)
def load_system_prompt(path: str | Path = SYSTEM_PROMPT) -> str:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("system", "").strip()


@lru_cache(maxsize=32)
def load_item_prompt(prompt_path: str) -> dict:
    """단일 항목 프롬프트 YAML 로드 (프로젝트 루트 기준 상대경로)."""
    path = Path(prompt_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def _load_few_shot(path: str | Path = FEW_SHOT_CONFIG) -> dict:
    if not Path(path).exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("examples", {})


def format_few_shot(item_id: int) -> str:
    """항목별 good/bad few-shot 예시를 LLM 가독 형태로 직렬화 (없으면 '')."""
    examples = _load_few_shot()
    ex = examples.get(item_id) or examples.get(str(item_id))
    if not ex:
        return ""
    blocks: list[str] = []
    for label, kor in (("good", "좋은 예"), ("bad", "나쁜 예")):
        case = ex.get(label)
        if not case:
            continue
        blocks.append(
            f"[{kor}] (점수 {case.get('score', '?')})\n"
            f"  스크립트: {str(case.get('transcript', '')).strip()}\n"
            f"  강점: {case.get('strengths', '')}\n"
            f"  개선점: {case.get('improvements', '')}"
        )
    return "\n".join(blocks)


def format_rag_context(chunks: list[IndexedChunk]) -> str:
    """RAG 검색 결과(청크)를 타임스탬프와 함께 직렬화."""
    if not chunks:
        return "(관련 발췌 없음)"
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[발췌 {i} | {c.start_timestamp}~{c.end_timestamp}]\n{c.text}")
    return "\n\n".join(parts)


def build_messages(
    item: ChecklistItem,
    chunks: list[IndexedChunk],
    bow: ItemBoW,
    document: LectureDocument | None = None,
    top_k: int | None = None,
) -> list[dict]:
    """LLM 메시지 리스트 생성: [{role: system}, {role: user}].

    google-genai 호출 측(analyzer)에서 system 은 system_instruction 으로,
    user 는 contents 로 변환한다.
    """
    prompt = load_item_prompt(item.prompt)
    user_template: str = prompt.get("user_template", "")

    fmt = _SafeDict(
        few_shot_examples=format_few_shot(item.id) or "(예시 없음)",
        rag_context=format_rag_context(chunks),
        positive_count=bow.positive_count,
        negative_count=bow.negative_count,
        bow_score=bow.bow_score,
        top_k=top_k if top_k is not None else len(chunks),
        filler_word_ratio=round(document.filler_word_ratio, 4) if document else "",
        avg_line_gap_seconds=round(document.avg_line_gap_seconds, 2) if document else "",
        completeness_rate=round(document.completeness_rate, 2) if document else "",
        consistency_ratio=round(document.consistency_ratio, 2) if document else "",
    )
    user_content = user_template.format_map(fmt).strip()

    return [
        {"role": "system", "content": load_system_prompt()},
        {"role": "user", "content": user_content},
    ]
