"""긴 강의 스크립트를 LLM 토큰 한도에 맞춰 청크 분할."""


def split(text: str, max_tokens: int = 4000, overlap: int = 200) -> list[str]:
    """토큰/문장 경계 기반 청크 분할 (placeholder)."""
    raise NotImplementedError
