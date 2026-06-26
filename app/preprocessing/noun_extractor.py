"""
Kiwi 형태소 분석 + 불용어 처리 모듈

utils.py → anchored_labeled.json 의 레코드를 받아
날짜 × 구간(개념/예시/실습)별 명사 후보 리스트를 반환한다.

주요 처리:
  1. COMPOUND_MERGE  : ASR 공백 분절 복합어 복원
  2. Kiwi 형태소 분석: NNG / NNP 명사만 추출
  3. 사용자 사전     : 복합 기술 용어 단일 토큰화
  4. STOPWORDS 필터  : 수동 불용어 (~170개)
  5. DF 자동 불용어  : 전체 날짜에 공통으로 등장하는 배경어 제거
                       단, GOLD_WHITELIST 단어는 보호
"""
from __future__ import annotations

import json
from collections import defaultdict, Counter
from pathlib import Path

from loguru import logger

# ── 경로 ──────────────────────────────────────────────────────────────────────
_BASE = Path(__file__).parent.parent.parent
USER_DICT_PATH = _BASE / "configs" / "kiwi_user_dict.yaml"

# ── 상수 ──────────────────────────────────────────────────────────────────────
MIN_WORD_LEN  = 2
LABEL_ORDER   = ["개념", "예시", "실습"]

# DF 자동 불용어에서 보호할 골드 핵심 용어
GOLD_WHITELIST = {"인덱스", "스트림", "버퍼", "트랜잭션"}

# ── 수동 불용어 ────────────────────────────────────────────────────────────────
STOPWORDS: set[str] = {
    # 담화어 · 메타어
    "다음", "이번", "지금", "경우", "내용", "부분", "생각", "나머지",
    "방법", "사용", "이유", "정도", "얼마", "중간", "이상", "이하",
    "의미", "형태", "단계", "번째", "관계", "기준", "역할", "목적",
    "이전", "이후", "처음", "마지막", "현재", "하단", "화면", "오후",
    "아침", "오늘", "내일", "시간", "확인", "설명", "진행", "작업",
    "얘기", "만약", "실제", "이해", "자체", "나중", "정리", "수업",
    "강사", "해당", "일반", "동일", "중요", "발생", "정확", "명시",
    "느낌", "눈도장", "마찬가지", "예전", "차이", "상황", "용어",
    "각각", "아무것", "기억", "단어", "기타", "가이드", "공유",
    "사람", "필요", "성공", "체크", "키워드", "버전", "형식",
    "오른쪽", "도움말", "섹션", "이용", "복사", "추가", "수정", "선택",
    # 복합어 분절 잔재 / ASR 음차 오류
    "바이", "언더", "네임", "마이에스큐",
    # ASR 전사 오류 — 02-02
    "스태틱", "어프", "인렬화", "오더링", "임플루먼트",
    "트리버셜", "코드이", "이노더링", "바이틀", "프리비지", "파일렛",
    # ASR 전사 오류 — 02-03 실습
    "됐쿠", "여어", "나네", "승제", "널까폴트", "스큐이", "후루",
    "엎퍼짓", "비언더버", "스티로", "화이리트",
    # ASR 전사 오류 — 02-03 예시/개념
    "캡세퍼라이터", "비짓", "스태팅", "닷넷", "애널리트", "디스크라이버",
    # 기타 메타/범용어
    "테스트", "스코어", "로고", "자식", "자식들",
    # 2차 ASR 오류
    "악푸스트림", "바이트스", "워팔트리", "보역", "트리뷰트",
    "객체야", "3트럴", "아까", "오브",
    # 범용 프로그래밍 메타어
    "처리", "추상", "병렬", "단위", "하위", "비트", "계산", "매개", "계열",
    # 02-03 개념 블로커
    "인메모리", "스크립트", "빅데이터",
    # 02-03 실습 블로커
    "명령", "명령어", "제어", "번호", "클로징",
    # 5차 rank check 기반
    "리드올바이트스", "바이티스", "리드올",
    "드로비", "림프스트리", "마이파일",
    "어트리뷰티스", "트루냐",
    "인터프라이즈", "고지서",
    "아스트리크", "마이에스켈", "원닝",
    "쭈르륵", "마이나니", "유즈", "사짝", "마이에케",
    "나지", "잠깐", "클라스", "심플", "크레에이트",
    "버퍼도",
    "직렬", "구현", "대상", "기본", "전위",
    "순차", "기반", "이름", "주요", "오류",
    "스텝", "총점", "이미지", "다운로드",
    "딜리트", "폴더", "디렉토리", "알고리즘",
    "프린트", "파일", "인풋",
    "전달", "커런트", "대소문자", "루트", "가감",
    "리턴", "리터럴", "원래", "리터럴리",
    "승계", "방향키", "신텍스", "넥스트",
    "한번", "헬프",
}

# ── ASR 공백 분절 복합어 복원 패턴 ────────────────────────────────────────────
COMPOUND_MERGE: list[tuple[str, str]] = [
    # 02-02 실습 골드
    ("파일 인풋 스트림",   "파일인풋스트림"),
    ("파일 아웃풋 스트림", "파일아웃풋스트림"),
    ("바이트 배열",        "바이트배열"),
    ("역 직렬화",          "역직렬화"),
    ("스탠다드 옵션",      "스탠다드옵션"),
    # 2-word 변형
    ("파일 인풋스트림",    "파일인풋스트림"),
    ("파일 아웃풋스트림",  "파일아웃풋스트림"),
    ("역직렬 화",          "역직렬화"),
    ("버퍼드 리더",        "버퍼드리더"),
    ("버퍼드 라이터",      "버퍼드라이터"),
    ("파일 리더",          "파일리더"),
    ("파일 라이터",        "파일라이터"),
    ("아웃풋 스트림",      "아웃풋스트림"),
    ("인풋 스트림",        "인풋스트림"),
    ("오브젝트 스트림",    "오브젝트스트림"),
    # 02-03 SQL
    ("슬로우 쿼리",        "슬로우쿼리"),
    ("바이너리 로그",      "바이너리로그"),
    ("에러 로그",          "에러로그"),
    ("마이 SQL",           "마이SQL"),
    ("이노 DB",            "이노DB"),
]


def normalize_text(text: str) -> str:
    for src, dst in COMPOUND_MERGE:
        text = text.replace(src, dst)
    return text


class NounExtractor:
    """
    Kiwi 형태소 분석 + 불용어 처리.

    Parameters
    ----------
    user_dict_path : Path
        kiwi_user_dict.txt 경로.
    df_auto_stopwords : bool
        전체 레코드를 순회해 DF=전체날짜인 단어를 자동 불용어로 추가할지 여부.
        True면 extract() 첫 호출 시 전체 레코드를 한 번 스캔.
    """

    def __init__(
        self,
        user_dict_path: Path = USER_DICT_PATH,
        df_auto_stopwords: bool = True,
    ):
        from kiwipiepy import Kiwi
        self._kiwi = Kiwi()
        self._load_user_dict(user_dict_path)
        self._df_auto = df_auto_stopwords
        self._stopwords = set(STOPWORDS)   # 인스턴스별 복사 (자동 불용어 추가용)

    def _load_user_dict(self, path: Path) -> None:
        if not path.exists():
            logger.warning(f"[NounExtractor] 사용자 사전 없음: {path}")
            return
        import yaml
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        count = 0
        for category, entry in data.items():
            if not isinstance(entry, dict):
                continue
            pos   = entry.get("pos", "NNG")
            words = entry.get("words") or []
            for word in words:
                self._kiwi.add_user_word(str(word), pos, 0.0)
                count += 1
        logger.info(f"[NounExtractor] 사용자 사전 {count}개 등록")

    def _add_df_stopwords(self, records: list[dict]) -> None:
        """DF=전체날짜 단어를 자동 불용어로 추가 (GOLD_WHITELIST 제외)."""
        doc_sets: dict[str, set[str]] = defaultdict(set)
        for rec in records:
            fname = rec.get("file", "")
            text = normalize_text(rec.get("text", ""))
            for tok in self._kiwi.tokenize(text):
                if tok.tag in ("NNG", "NNP") and len(tok.form) >= MIN_WORD_LEN:
                    doc_sets[fname].add(tok.form)

        n_docs = len(doc_sets)
        if n_docs == 0:
            return

        df_counter: Counter[str] = Counter()
        for ws in doc_sets.values():
            for w in ws:
                df_counter[w] += 1

        auto = {w for w, cnt in df_counter.items() if cnt >= n_docs} - GOLD_WHITELIST
        self._stopwords.update(auto)
        logger.info(
            f"[NounExtractor] 자동 불용어 추가 {len(auto)}개 "
            f"(DF={n_docs}, 총 불용어 {len(self._stopwords)}개)"
        )

    def extract_from_text(self, text: str) -> list[str]:
        """단일 텍스트에서 명사 리스트 반환 (중복 제거, 등장 순서 유지)."""
        text = normalize_text(text)
        seen: dict[str, None] = {}
        for tok in self._kiwi.tokenize(text):
            if (
                tok.tag in ("NNG", "NNP")
                and len(tok.form) >= MIN_WORD_LEN
                and tok.form not in self._stopwords
            ):
                seen[tok.form] = None
        return list(seen)

    def extract(
        self,
        records: list[dict],
    ) -> dict[tuple[str, str], list[str]]:
        """
        레코드 리스트 전체에서 날짜 × 구간별 명사 후보를 추출한다.

        Parameters
        ----------
        records : list[dict]
            anchored_labeled.json 레코드 (file, llm_label, text 키 필요).

        Returns
        -------
        dict[(date, label), list[str]]
            {("2026-02-02", "개념"): ["직렬화", "스트림", ...], ...}
        """
        if self._df_auto:
            self._add_df_stopwords(records)

        # 날짜 × 레이블별 텍스트 누적
        grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
        for rec in records:
            fname = rec.get("file", "")
            # "2026-02-02_kdt-backendj-21th.txt" → "2026-02-02"
            date = fname.split("_")[0] if fname else rec.get("date", "")
            label = rec.get("llm_label", "기타")
            if label in LABEL_ORDER:
                grouped[(date, label)].append(rec.get("text", ""))

        result: dict[tuple[str, str], list[str]] = {}
        for (date, label), texts in sorted(grouped.items()):
            combined = " ".join(texts)
            nouns = self.extract_from_text(combined)
            result[(date, label)] = nouns
            logger.debug(
                f"[NounExtractor] {date} [{label}] → 명사 {len(nouns)}개"
            )

        return result


# ── 편의 함수 ─────────────────────────────────────────────────────────────────

def extract_nouns_from_json(
    json_path: str | Path,
    user_dict_path: Path = USER_DICT_PATH,
    df_auto_stopwords: bool = True,
) -> dict[tuple[str, str], list[str]]:
    """
    anchored_labeled.json 파일에서 바로 명사 후보를 추출한다.

    Returns
    -------
    {("2026-02-02", "개념"): ["직렬화", ...], ...}
    """
    records = json.loads(Path(json_path).read_text(encoding="utf-8"))
    extractor = NounExtractor(
        user_dict_path=user_dict_path,
        df_auto_stopwords=df_auto_stopwords,
    )
    return extractor.extract(records)
