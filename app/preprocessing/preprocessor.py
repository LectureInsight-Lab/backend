"""STT 파싱 + 세션/구간 분리 (1단계).

STT 포맷: ``<HH:MM:SS> speaker_id: 발화 텍스트``

특이점 (실데이터 관찰):
- 타임스탬프는 **12시간 cyclic** 표기 (AM/PM 표기 없음).
  시(hour) 값이 갑자기 감소하면 PM 전환으로 보고 +12 보정.
- 한 파일에 오전(09:00~12:00) + 오후(13:00~18:00) 세션이 합쳐져 있음.
  발화 갭이 30분 이상이면 새 세션으로 분리.
- 강사 마이크 위주로 녹음되어 화자 분리(강사/학생)는 사실상 불필요.
  여러 speaker_id 가 보이는 파일은 세션 경계에서 STT 가 id 를 재생성한 경우가 대부분.

산출물:
- ``Utterance`` 리스트 → 세션 묶음 → intro/middle/outro 슬라이스
- ``LectureDocument`` 에는 세션 단위 + 파일 평탄화 두 버전을 모두 채움.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

from app.analysis.schemas import LectureDocument, Session, Utterance
from app.core.paths import read_stt
from app.preprocessing import eda

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKLIST_CONFIG = PROJECT_ROOT / "configs" / "checklist.yaml"

# 줄 포맷: ``<HH:MM:SS> speaker_id: text``
_LINE_RE = re.compile(r"^<(\d{2}):(\d{2}):(\d{2})>\s*([^:]+?):\s*(.*)$")

SESSION_GAP_SECONDS = 30 * 60        # 30분 이상 갭 → 새 세션
MIN_SESSION_LINES = 30               # 그보다 적으면 STT 꼬리 노이즈로 보고 제외
ROLLOVER_THRESHOLD_SECONDS = 60 * 60  # 1시간 이상 역행 → PM 전환
SESSION_LABELS = ("morning", "afternoon", "evening")


# ── 1) STT 텍스트 → Utterance 리스트 ───────────────────────────
def parse(raw_text: str) -> list[Utterance]:
    """원본 STT를 Utterance 리스트로 파싱.

    - 정규식 미매칭 라인은 무시 (헤더/빈 줄 등).
    - 12h cyclic → 24h 변환: 시 값이 1시간 넘게 역행하면 +12h 누적.
    - ``seconds_from_start`` 는 파일 첫 발화 기준 경과 초.
    """
    utterances: list[Utterance] = []
    hour_offset = 0           # 12h cyclic 보정용 누적 (초)
    prev_real_total: int | None = None
    base_total: int | None = None

    for raw_line in raw_text.splitlines():
        m = _LINE_RE.match(raw_line.strip())
        if not m:
            continue
        hh, mm, ss = int(m.group(1)), int(m.group(2)), int(m.group(3))
        speaker = m.group(4).strip()
        text = m.group(5).strip()
        if not text:
            continue

        cur_in_cycle = hh * 3600 + mm * 60 + ss
        cur_real = cur_in_cycle + hour_offset

        # 큰 폭의 역행이 보이면 PM 전환으로 간주
        if prev_real_total is not None and cur_real < prev_real_total - ROLLOVER_THRESHOLD_SECONDS:
            hour_offset += 12 * 3600
            cur_real = cur_in_cycle + hour_offset

        if base_total is None:
            base_total = cur_real
        prev_real_total = cur_real

        seconds_from_start = cur_real - base_total
        norm_hh = (cur_real // 3600) % 24
        norm_mm = (cur_real % 3600) // 60
        norm_ss = cur_real % 60

        utterances.append(
            Utterance(
                timestamp=f"{norm_hh:02d}:{norm_mm:02d}:{norm_ss:02d}",
                speaker_id=speaker,
                text=text,
                seconds_from_start=seconds_from_start,
            )
        )
    return utterances


# ── 2) 세션 분리 (점심 갭 기준) ────────────────────────────────
def split_sessions(
    utterances: list[Utterance],
    gap_threshold_seconds: int = SESSION_GAP_SECONDS,
    min_session_lines: int = MIN_SESSION_LINES,
) -> list[list[Utterance]]:
    """발화 간 갭이 임계값 이상이면 새 세션으로 분리.

    분리 후 줄 수가 ``min_session_lines`` 미만인 세션은 노이즈로 보고 제외.
    (예: 17:50 에 4줄짜리 인사 마무리만 잡힌 STT 꼬리)
    """
    if not utterances:
        return []
    groups: list[list[Utterance]] = [[utterances[0]]]
    for prev, cur in zip(utterances, utterances[1:]):
        if cur.seconds_from_start - prev.seconds_from_start >= gap_threshold_seconds:
            groups.append([cur])
        else:
            groups[-1].append(cur)
    return [g for g in groups if len(g) >= min_session_lines]


# ── 3) 세션 내 intro/middle/outro 슬라이스 ────────────────────
def split_segments(
    session_lines: list[Utterance],
    intro_minutes: int = 20,
    outro_minutes: int = 15,
) -> tuple[list[Utterance], list[Utterance], list[Utterance]]:
    """단일 세션을 (intro, middle, outro) 로 자른다.

    - intro: 세션 시작 후 ``intro_minutes`` 이내 발화
    - outro: 세션 종료 전 ``outro_minutes`` 이내 발화
    - middle: 그 사이

    세션이 너무 짧아 intro 와 outro 구간이 겹치면 middle 은 빈 리스트.
    """
    if not session_lines:
        return [], [], []
    start = session_lines[0].seconds_from_start
    end = session_lines[-1].seconds_from_start
    intro_cutoff = start + intro_minutes * 60
    outro_cutoff = end - outro_minutes * 60

    intro = [u for u in session_lines if u.seconds_from_start <= intro_cutoff]
    outro = [u for u in session_lines if u.seconds_from_start >= outro_cutoff]
    middle = [
        u for u in session_lines
        if intro_cutoff < u.seconds_from_start < outro_cutoff
    ]
    return intro, middle, outro


# ── 4) checklist.yaml 의 segments 설정 로드 ───────────────────
def load_segment_config(path: Path = CHECKLIST_CONFIG) -> tuple[int, int]:
    """``configs/checklist.yaml`` 에서 intro/outro 분 단위 로드."""
    if not path.exists():
        return 20, 15
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    seg = raw.get("segments", {})
    return int(seg.get("intro_minutes", 20)), int(seg.get("outro_minutes", 15))


# ── 5) raw_text → LectureDocument 통합 ────────────────────────
def build_document(
    raw_text: str,
    lecture_date: str,
    instructor_id: str,
    intro_minutes: int | None = None,
    outro_minutes: int | None = None,
) -> LectureDocument:
    """STT 원문을 통째로 전처리해 ``LectureDocument`` 반환."""
    if intro_minutes is None or outro_minutes is None:
        cfg_intro, cfg_outro = load_segment_config()
        intro_minutes = intro_minutes if intro_minutes is not None else cfg_intro
        outro_minutes = outro_minutes if outro_minutes is not None else cfg_outro

    utterances = parse(raw_text)
    session_groups = split_sessions(utterances)

    sessions: list[Session] = []
    for idx, group in enumerate(session_groups):
        intro, middle, outro = split_segments(group, intro_minutes, outro_minutes)
        label = SESSION_LABELS[idx] if idx < len(SESSION_LABELS) else f"session_{idx + 1}"
        sessions.append(
            Session(
                session_index=idx,
                label=label,
                start_timestamp=group[0].timestamp,
                end_timestamp=group[-1].timestamp,
                all_lines=group,
                intro_lines=intro,
                middle_lines=middle,
                outro_lines=outro,
            )
        )

    flat_intro = [u for s in sessions for u in s.intro_lines]
    flat_middle = [u for s in sessions for u in s.middle_lines]
    flat_outro = [u for s in sessions for u in s.outro_lines]

    return LectureDocument(
        lecture_date=lecture_date,
        instructor_id=instructor_id,
        all_lines=utterances,
        sessions=sessions,
        intro_lines=flat_intro,
        middle_lines=flat_middle,
        outro_lines=flat_outro,
        filler_word_ratio=eda.filler_word_ratio(utterances),
        avg_line_gap_seconds=eda.avg_line_gap_seconds(utterances),
    )


def build_document_from_paths(lecture_date: str, course_id: str, instructor_id: str) -> LectureDocument:
    """``configs/paths.yaml`` 의 stt_dir 에서 파일을 찾아 빌드."""
    raw_text = read_stt(lecture_date, course_id)
    return build_document(raw_text, lecture_date=lecture_date, instructor_id=instructor_id)
