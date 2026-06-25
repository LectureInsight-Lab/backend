"""학생 질문-응답 페어 추출.

question_eval_eda.ipynb 에서 확정된 로직 이식.
LectureDocument.all_lines 를 받아 Q-A 페어 목록(JSON)을 반환한다.

탐지 방식:
    - 학생이 강사에게 묻는 명시적 질문 패턴 regex 매칭
    - 제외 패턴(다른 강사님 등) 필터링
    - 질문 이후 최대 30문장을 답변으로 수집 (다음 질문 등장 시 중단)
"""

import json
import re
from pathlib import Path

import kss
import pandas as pd

from app.analysis.schemas import Utterance

# ─── 탐지 패턴 ──────────────────────────────────────────────────

_QUESTION_PATTERNS = [
    # 강사님 + 격식 어미
    r"강사님.{0,80}(해야\s*돼요|해야\s*되나요|해도\s*돼요|해도\s*되나요|써도\s*돼요|써도\s*되나요)",
    r"강사님.{0,80}(되나요|될까요|돼요|안\s*돼요|안\s*되나요|안\s*되는데요)",
    r"강사님.{0,80}(뭐예요|뭔가요|뭐죠|어떻게\s*해요|어떻게\s*하나요|왜\s*그래요|어디\s*있어요)",
    r"강사님.{0,80}(에러|오류|워닝|안\s*나와요|안\s*떠요)",
    # 강사님 + 구어체 어미
    r"강사님.{0,80}(이게\s*뭐야|뭐야|뭔데|거야|건가요|건데요)",
    r"강사님.{0,80}(어떡해|어쩌지|어쩌죠)",
    r"강사님.{0,80}(넣어봐도|해봐도|써봐도|봐도\s*돼)",
    r"강사님.{0,80}(하고\s*싶어|넣고\s*싶어|쓰고\s*싶어|하고\s*싶은데)",
    r"강사님.{0,80}없어요",
    r"강사님.{0,80}나요",
    r"강사님.{0,80}(지\s*않아|않아요|않는데요)",
    # 후치 강사님 (질문어 앞, 강사님 뒤)
    r"(어떻게|어디|왜).{0,80}강사님(?!이|들)",
    # 질문 명시
    r".{0,80}질문이\s*(있는데|들어왔는데|나왔는데|왔는데)",
    r".{0,80}질문을\s*(주셨는데|하셨는데|했는데)",
    r".{0,80}라고\s*질문",
    r".{0,80}라는\s*질문",
    r".{0,80}물어보셨는데",
    r".{0,80}물어보시는데",
    r".{0,80}여쭤보셨는데",
    r".{0,80}여쭤보시는데",
    r"(채팅|채팅창|DM|디엠).{0,50}(질문|올라왔|남겨|보내|주셨|주셨어)",
]

_EXCLUDE_PATTERNS = [
    r"다른\s*강사님",
    r"나는\s*강사님",
]

_ANSWER_WINDOW = 30   # 답변으로 수집할 최대 문장 수
_MIN_TEXT_LEN  = 8    # KSS 입력 전 발화 최소 길이 (노이즈 제거)

_q_re  = re.compile("|".join(_QUESTION_PATTERNS))
_ex_re = re.compile("|".join(_EXCLUDE_PATTERNS))


def _is_question(text: str) -> bool:
    if _ex_re.search(text):
        return False
    return bool(_q_re.search(text))


def build_qa_llm_payload(
    utterances: list[Utterance],
    answer_window: int = _ANSWER_WINDOW,
) -> list[dict]:
    """utterances → Q-A 페어 JSON 리스트.

    Args:
        utterances:    LectureDocument.all_lines (전체 발화)
        answer_window: 답변으로 수집할 최대 문장 수

    Returns:
        [
            {
                "question":             str,
                "answer":               str,
                "answer_sentence_count": int,
            },
            ...
        ]
    """
    full_text = " ".join(u.text for u in utterances if len(u.text) >= _MIN_TEXT_LEN)
    sentences = kss.split_sentences(full_text, backend="fast")

    pairs = []
    for i, sent in enumerate(sentences):
        if not _is_question(sent):
            continue

        answer_sents = []
        for s in sentences[i + 1 : i + 1 + answer_window]:
            if _is_question(s):
                break
            answer_sents.append(s)

        pairs.append({
            "question": sent,
            "answer": " ".join(answer_sents),
            "answer_sentence_count": len(answer_sents),
        })

    return pairs


def run(df: pd.DataFrame) -> dict:
    """단일 강의 DataFrame → Q-A 페어 추출.

    Args:
        df: 단일 강의 발화 DataFrame (텍스트 파일 1개분).
            컬럼 기준: timestamp, speaker_id, text_raw

    Returns:
        {"chunk": str} — 전체 Q-A 페어를 하나의 문자열로 합친 dict.
    """
    text_col = "text_raw" if "text_raw" in df.columns else "text"
    utterances = [
        Utterance(
            timestamp=str(row.get("timestamp", "")),
            speaker_id=str(row.get("speaker_id", "")),
            text=str(row.get(text_col, "")),
            seconds_from_start=0,
        )
        for _, row in df.iterrows()
    ]
    pairs = build_qa_llm_payload(utterances)
    chunk = "\n\n".join(f"Q: {p['question']}\nA: {p['answer']}" for p in pairs)
    return {"chunk": chunk}


# ── CLI 진입점 ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    from app.preprocessing.utils import parse_and_split

    parser = argparse.ArgumentParser(description="학생 질문-응답 페어 추출")
    parser.add_argument("txt_path", help="원본 강의 텍스트 파일 경로 (단일 강의)")
    parser.add_argument("--output", "-o", default=None,  help="결과 JSON 저장 경로 (생략 시 stdout)")
    args = parser.parse_args()

    _df = parse_and_split(args.txt_path)
    result = run(_df)

    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"저장 완료: {args.output}")
    else:
        print(output)
