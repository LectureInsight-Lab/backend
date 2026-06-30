"""
키워드 강조 여부 확인 모듈

KeywordExtractor가 반환한 키워드 리스트를 받아
각 키워드가 raw STT 텍스트에서 실제로 강조되고 있는지 확인한다.

탐지 신호 두 가지:
  1. 명시적 강조 마커 (regex)
     — 중요합니다, 꼭 기억, 핵심은, 반드시 … 등
     — 키워드 등장 발화 ±WINDOW_UTTERS 줄 내에 마커가 있으면 강조로 인정
  2. 반복 강조
     — REPEAT_WINDOW(초) 내에 같은 키워드가 REPEAT_COUNT회 이상 등장
"""
from __future__ import annotations

import re
from pathlib import Path
from collections import defaultdict

from loguru import logger

from app.preprocessing.item07_noun_extractor import normalize_text

# ── 파라미터 ──────────────────────────────────────────────────────────────────
WINDOW_UTTERS = 3    # 키워드 발화 앞뒤 탐색 반경 (발화 단위)
REPEAT_WINDOW = 300  # 반복 강조 판정 시간 창 (초, 5분)
REPEAT_COUNT  = 3    # 이 횟수 이상이면 반복 강조로 인정

_LINE_RE = re.compile(r"^<(\d{2}:\d{2}:\d{2})>\s+\S+:\s*(.*)$")

# ── 강조 마커 regex ────────────────────────────────────────────────────────────
_EMPHASIS_RE = re.compile(
    # 중요 계열 (STT에서 '것'→'거' 음운변환 포함)
    r"중요합니다|중요해요|중요해\b|중요하게|중요한\s*(것|거|포인트|개념|부분|내용)"
    r"|가장\s*중요한"
    r"|이게\s*중요|여기서\s*중요|이\s*부분이\s*중요"
    # 기억/숙지 요청
    r"|꼭\s*(기억|알아야|봐야|확인|외워|이해)"
    r"|반드시\s*(기억|알아야|봐야|이해|알고)"
    r"|기억해\s*(두세요|두셔야|주세요|하세요)"
    r"|기억하셔야|꼭\s*기억하세요"
    r"|기억하고\s*계셔야"
    # 핵심/포인트
    r"|핵심은|핵심이|핵심\s*개념|핵심\s*포인트|포인트는|포인트가"
    # 강조 행위 직접 언급
    r"|강조하고\s*싶|강조드리|강조해서|강조를\s*하고"
    r"|밑줄\s*(그어|쳐|치세요|그으세요)|눈도장"
    # 반복 명시
    r"|다시\s*한\s*(번|번만)\s*(말씀|설명|강조|얘기)"
    r"|한\s*번\s*더\s*(말씀|설명|강조|얘기|볼)"
    # 주목 유도
    r"|자\s*여기서\s*잘|여기\s*잘\s*봐|집중해서\s*봐"
    r"|이거\s*진짜\s*(중요|봐야|기억)"
    # 알고 가 (take-away 강조: "알고 가자", "알고 가셔야", "알고 가야 돼")
    r"|알고\s+(가셔야|가야|가자|가세요|계셔야)"
    # 주의 신호 ("주의할 점은", "주의할 것")
    r"|주의할\s*(점|것|거)"
    # 알아야/있어야 계열 (FN 보완: "알아야 됩니다", "알아야 돼", "알고 있어야")
    r"|알아야\s*(되는|돼|됩|된다|되고|합니다)"
    r"|알고\s+있어야"
    # 확인/점검 필수 (FN 보완: "확인해야 된다")
    r"|확인해야\s*(된다|돼|됩)"
    # 시험 출제 언급 (FN 보완: "많이 틀리는데" — 빈출 오답 강조)
    r"|많이\s*틀리"
    # 정리/요약 강조 (FN 보완: "정리해드리는데", "정리 리플레이스는")
    r"|정리\s*해드리"
)


def _ts_to_sec(ts: str) -> int:
    h, m, s = map(int, ts.split(":"))
    return h * 3600 + m * 60 + s


def _parse_stt(txt_path: Path) -> list[tuple[int, str]]:
    """
    raw STT 파일 → (elapsed_sec, text) 리스트.
    12시간 시계 역행(오전→오후 전환)을 running_sec delta로 보정.
    """
    parsed: list[tuple[str, str]] = []
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        m = _LINE_RE.match(line.strip())
        if m:
            parsed.append((m.group(1), m.group(2).strip()))

    if not parsed:
        return []

    result: list[tuple[int, str]] = []
    running_sec = 0
    prev_abs = _ts_to_sec(parsed[0][0])

    for ts, text in parsed:
        abs_sec = _ts_to_sec(ts)
        delta = abs_sec - prev_abs
        if delta < -3600:    # 12시간 시계 역행 보정
            delta += 12 * 3600
        elif delta < 0:
            delta = 0
        running_sec += delta
        prev_abs = abs_sec
        if text:
            result.append((running_sec, normalize_text(text)))

    return result


class EmphasisChecker:
    """
    키워드별 강조 여부를 탐지한다.

    Parameters
    ----------
    window_utters : int
        키워드 발화 기준 앞뒤 몇 줄까지 마커를 탐색할지.
    repeat_window : int
        반복 강조 판정 시간 창 (초).
    repeat_count : int
        반복 강조 최소 횟수.
    """

    def __init__(
        self,
        window_utters: int = WINDOW_UTTERS,
        repeat_window: int = REPEAT_WINDOW,
        repeat_count:  int = REPEAT_COUNT,
    ):
        self._win   = window_utters
        self._rwin  = repeat_window
        self._rcnt  = repeat_count

    def check_keyword(
        self,
        keyword: str,
        utters: list[tuple[int, str]],
    ) -> tuple[bool, str]:
        """
        단일 키워드에 대한 강조 신호 탐지.

        Returns
        -------
        (강조 여부, 근거 문자열)
        """
        hit_indices = [i for i, (_, t) in enumerate(utters) if keyword in t]

        if not hit_indices:
            return False, "텍스트 미등장"

        # 신호 1: 명시적 강조 마커
        for idx in hit_indices:
            w_start = max(0, idx - self._win)
            w_end   = min(len(utters), idx + self._win + 1)
            window_text = " ".join(t for _, t in utters[w_start:w_end])
            m = _EMPHASIS_RE.search(window_text)
            if m:
                return True, f"마커({m.group()[:20]}) @ 발화#{idx}"

        # 신호 2: 반복 강조
        hit_secs = [utters[i][0] for i in hit_indices]
        for sec in hit_secs:
            window = [s for s in hit_secs if sec <= s < sec + self._rwin]
            if len(window) >= self._rcnt:
                return True, f"반복강조({len(window)}회/{self._rwin//60}분)"

        return False, f"등장{len(hit_indices)}회·강조미확인"

    def check(
        self,
        keywords: list[str],
        txt_path: str | Path,
    ) -> dict[str, tuple[bool, str]]:
        """
        키워드 리스트 전체에 대해 강조 여부를 확인한다.

        Parameters
        ----------
        keywords : list[str]
            KeywordExtractor 에서 받은 키워드 리스트 (단어만, 점수 제외).
        txt_path : str | Path
            raw STT .txt 파일 경로.

        Returns
        -------
        {keyword: (강조 여부, 근거)}
        """
        utters = _parse_stt(Path(txt_path))
        if not utters:
            logger.warning(f"[EmphasisChecker] STT 파싱 실패: {txt_path}")
            return {kw: (False, "STT 파싱 실패") for kw in keywords}

        result: dict[str, tuple[bool, str]] = {}
        for kw in keywords:
            emphasized, reason = self.check_keyword(kw, utters)
            result[kw] = (emphasized, reason)
            marker = "✓" if emphasized else "✗"
            logger.debug(f"[EmphasisChecker] {marker} {kw}: {reason}")

        return result

    def summarize(
        self,
        emphasis_result: dict[str, tuple[bool, str]],
    ) -> dict:
        """
        check() 결과를 요약한다.

        Returns
        -------
        {
          "emphasized": ["직렬화", ...],
          "not_emphasized": ["채널", ...],
          "rate": 0.6,           # 강조 비율
          "details": [str, ...], # "✓ 직렬화: 마커(...)" 형식
        }
        """
        emphasized     = [kw for kw, (ok, _) in emphasis_result.items() if ok]
        not_emphasized = [kw for kw, (ok, _) in emphasis_result.items() if not ok]
        total = len(emphasis_result)
        rate  = len(emphasized) / total if total else 0.0

        details = [
            f"{'✓' if ok else '✗'} {kw}: {reason}"
            for kw, (ok, reason) in emphasis_result.items()
        ]

        return {
            "emphasized":     emphasized,
            "not_emphasized": not_emphasized,
            "rate":           round(rate, 3),
            "details":        details,
        }


# ── 편의 함수 ─────────────────────────────────────────────────────────────────

def check_emphasis(
    keywords: list[str],
    txt_path: str | Path,
) -> dict:
    """
    EmphasisChecker().check() + summarize() 를 한 번에 실행하는 단축 함수.

    Returns
    -------
    summarize() 반환 딕셔너리
    """
    checker = EmphasisChecker()
    result  = checker.check(keywords, txt_path)
    return checker.summarize(result)
