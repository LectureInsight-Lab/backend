"""프롬프트 조립기 (Paper #2, #6).

항목별 평가 기준(루브릭) + Few-shot 예시 + RAG 컨텍스트 + BoW 카운트
를 합쳐 LLM에 보낼 메시지를 만든다.

구조:
    [시스템] 루브릭 정의 (1~5점 단계별 의미)
    [유저]   ① 항목 정의 및 평가 기준
            ② Few-shot 예시 (good + bad)        ← Paper #2
            ③ RAG 검색 결과 (top-K 청크)         ← Paper #5
            ④ BoW 카운트                       ← Paper #3
            ⑤ 출력 형식 (JSON 강제)

TODO:
- prompts/system.yaml + items/*.yaml 로더
- few_shot_examples.yaml 통합
- Jinja2 또는 str.format 기반 템플릿 엔진
"""
from app.analysis.schemas import IndexedChunk, ItemBoW
from app.core.checklist import ChecklistItem


def build_messages(
    item: ChecklistItem,
    chunks: list[IndexedChunk],
    bow: ItemBoW,
) -> list[dict]:
    """OpenAI Chat Completion 메시지 리스트 생성 (placeholder).

    반환 예시::
        [
            {"role": "system", "content": "<루브릭 정의>"},
            {"role": "user",   "content": "<항목 + 예시 + 컨텍스트 + BoW>"},
        ]
    """
    raise NotImplementedError


def load_item_prompt(prompt_path: str) -> dict:
    """단일 항목 프롬프트 YAML 로드 (placeholder)."""
    raise NotImplementedError


def format_rag_context(chunks: list[IndexedChunk]) -> str:
    """RAG 검색 결과를 LLM이 읽기 쉬운 형태로 직렬화 (placeholder)."""
    raise NotImplementedError
