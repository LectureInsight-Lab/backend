"""preprocessor + eda 회귀 테스트.

두 케이스를 검증:
- 02-02: 단일 세션, 단일 speaker_id (b54f46b0)
- 02-13: 이중 세션, 오후 세션이 12h cyclic 으로 표기됨 (09→10→11 후 01~)
"""
from __future__ import annotations

import pytest

from app.core.paths import load_paths, read_stt
from app.preprocessing import eda, preprocessor


# ── 합성 데이터 단위 테스트 ─────────────────────────────────
def test_parse_basic_format():
    raw = (
        "<09:00:00> abc: 안녕하세요\n"
        "<09:00:05> abc: 오늘 수업 시작합니다\n"
    )
    utterances = preprocessor.parse(raw)
    assert len(utterances) == 2
    assert utterances[0].timestamp == "09:00:00"
    assert utterances[0].speaker_id == "abc"
    assert utterances[0].text == "안녕하세요"
    assert utterances[0].seconds_from_start == 0
    assert utterances[1].seconds_from_start == 5


def test_parse_12h_cyclic_rollover():
    """11:55 → 01:00 은 PM 전환 → 12 추가."""
    raw = (
        "<09:00:00> a: 시작\n"
        "<11:55:00> a: 오전 끝\n"
        "<01:00:00> a: 오후 시작\n"
        "<05:50:00> a: 오후 끝\n"
    )
    utterances = preprocessor.parse(raw)
    assert [u.seconds_from_start for u in utterances] == [
        0,
        (11 - 9) * 3600 + 55 * 60,         # 02:55:00 경과
        (13 - 9) * 3600,                    # 04:00:00 경과
        (17 - 9) * 3600 + 50 * 60,          # 08:50:00 경과
    ]
    # 13시 라인은 timestamp 가 24h 표기로 정규화
    assert utterances[2].timestamp == "13:00:00"
    assert utterances[3].timestamp == "17:50:00"


def test_parse_skips_unparseable_lines():
    raw = (
        "=== Header ===\n"
        "<09:00:00> a: 본문\n"
        "\n"
        "trailing junk\n"
    )
    utterances = preprocessor.parse(raw)
    assert len(utterances) == 1
    assert utterances[0].text == "본문"


def test_split_sessions_by_gap():
    # 오전 세션과 오후 세션 각각 30줄 이상 + 점심 갭은 60분 이상.
    morning_lines = [f"<09:{m // 60:02d}:{m % 60:02d}> a: 오전{m}\n" for m in range(0, 60, 1)]   # 09:00~09:59
    afternoon_lines = [f"<01:{m // 60:02d}:{m % 60:02d}> a: 오후{m}\n" for m in range(0, 60, 1)]  # 13:00~13:59
    raw = "".join(morning_lines) + "".join(afternoon_lines)
    utterances = preprocessor.parse(raw)
    sessions = preprocessor.split_sessions(utterances)
    assert len(sessions) == 2
    assert [len(s) for s in sessions] == [60, 60]


def test_split_sessions_drops_tiny_trailer():
    """세션 분리 후 30줄 미만짜리 꼬리 세션은 노이즈로 보고 제외."""
    morning_lines = [f"<09:{m // 60:02d}:{m % 60:02d}> a: 본문{m}\n" for m in range(0, 60, 1)]
    trailer = "<05:50:00> a: 수고하셨습니다\n<05:50:05> a: 감사합니다\n"   # 17:50, 2줄
    raw = "".join(morning_lines) + trailer
    utterances = preprocessor.parse(raw)
    sessions = preprocessor.split_sessions(utterances)
    assert len(sessions) == 1
    assert len(sessions[0]) == 60   # trailer 제외


def test_split_segments_short_session_overlap_ok():
    raw = (
        "<09:00:00> a: line1\n"
        "<09:05:00> a: line2\n"
        "<09:10:00> a: line3\n"
    )
    utterances = preprocessor.parse(raw)
    # 10분짜리 세션에 intro 20분 + outro 15분 → 다 겹침
    intro, middle, outro = preprocessor.split_segments(utterances, intro_minutes=20, outro_minutes=15)
    assert len(intro) == 3
    assert len(outro) == 3
    assert middle == []


# ── EDA ─────────────────────────────────────────────────
def test_filler_word_ratio_counts_exact_token():
    raw = (
        "<09:00:00> a: 어 음 안녕하세요\n"        # 어, 음, 안녕하세요 → 2/3
        "<09:00:05> a: 그래서 좋아요\n"           # '그래서' 는 '그' 와 다름 → 0/2
    )
    utterances = preprocessor.parse(raw)
    ratio = eda.filler_word_ratio(utterances)
    # 총 5 어절 중 2개가 필러
    assert ratio == pytest.approx(2 / 5)


def test_avg_line_gap_excludes_large_gaps():
    raw = (
        "<09:00:00> a: l1\n"
        "<09:00:10> a: l2\n"
        "<11:55:00> a: l3\n"
        "<01:00:00> a: l4\n"     # 65분 갭 → 제외
        "<01:00:30> a: l5\n"
    )
    utterances = preprocessor.parse(raw)
    avg = eda.avg_line_gap_seconds(utterances)
    # 갭: 10s, (11:55 - 09:00:10) = 10490s, (제외), 30s
    # 30분(=1800s) 이상은 제외 → 10490 도 제외
    # 남는 것: 10, 30 → 평균 20
    assert avg == pytest.approx(20.0)


def test_basic_stats_shape():
    raw = (
        "<09:00:00> a: 하나 둘 셋\n"
        "<09:00:05> b: 하나 넷\n"
    )
    utterances = preprocessor.parse(raw)
    stats = eda.basic_stats(utterances)
    assert stats["line_count"] == 2
    assert stats["token_count"] == 5
    assert stats["unique_token_count"] == 4  # 하나/둘/셋/넷
    assert stats["speaker_count"] == 2


# ── 실데이터 회귀 테스트 (configs/paths.yaml 필요) ─────────
def _real_data_available() -> bool:
    try:
        cfg = load_paths()
    except FileNotFoundError:
        return False
    return cfg.data.stt_dir.exists()


pytestmark_real = pytest.mark.skipif(
    not _real_data_available(),
    reason="configs/paths.yaml 의 stt_dir 가 없습니다.",
)


@pytestmark_real
def test_real_02_02_single_session():
    raw = read_stt("2026-02-02", "kdt-backendj-21th")
    doc = preprocessor.build_document(raw, lecture_date="2026-02-02", instructor_id="kim_youngah")
    assert len(doc.all_lines) > 1000
    # 단일 화자
    speakers = {u.speaker_id for u in doc.all_lines}
    assert speakers == {"b54f46b0"}
    # 꼬리 노이즈 필터 적용 후 오전/오후 2개 세션
    assert len(doc.sessions) == 2
    assert doc.sessions[0].label == "morning"
    assert doc.sessions[1].label == "afternoon"
    # 보조 통계 합리 범위
    assert 0.0 <= doc.filler_word_ratio < 0.5
    assert doc.avg_line_gap_seconds > 0


@pytestmark_real
def test_real_02_13_two_sessions_and_pm_rollover():
    raw = read_stt("2026-02-13", "kdt-backendj-21th")
    doc = preprocessor.build_document(raw, lecture_date="2026-02-13", instructor_id="kim_youngah")
    # 17:50 짜리 4줄 인사 꼬리는 제외 → 오전 + 오후 2세션
    assert len(doc.sessions) == 2
    assert doc.sessions[0].label == "morning"
    assert doc.sessions[1].label == "afternoon"
    # PM 전환 검증: 마지막 발화의 seconds_from_start 가 충분히 큰 값
    last = doc.all_lines[-1]
    assert last.seconds_from_start > 6 * 3600   # 6시간 이상 진행
    # 마지막 timestamp 는 24h 표기로 17~18시대
    last_hh = int(last.timestamp.split(":")[0])
    assert 13 <= last_hh <= 23


# ── with_sentences 배선 (항목 2·3 핸드오프) ─────────────────
def test_build_document_without_sentences_is_lean():
    """기본(with_sentences=False)은 문장화 미수행 → 신규 필드 기본값."""
    raw = "<09:00:00> a: 오늘은 자바를 배웁니다\n<09:00:05> a: 그래서 시작합니다\n"
    doc = preprocessor.build_document(raw, lecture_date="2026-02-02", instructor_id="kim")
    assert doc.sentences == []
    assert doc.completeness_rate == 0.0
    assert doc.consistency_ratio == 0.0
    assert doc.violation_count == 0


def test_build_document_with_sentences_populates():
    """with_sentences=True → sentences/completeness_rate/consistency_ratio 채워짐."""
    pytest.importorskip("kiwipiepy")
    raw = "<09:00:00> a: 오늘은 자바를 배웁니다\n<09:00:05> a: 그래서 이게 중요하고\n"
    doc = preprocessor.build_document(
        raw, lecture_date="2026-02-02", instructor_id="kim", with_sentences=True
    )
    assert len(doc.sentences) >= 1
    assert 0.0 <= doc.completeness_rate <= 100.0
    assert 0.0 <= doc.consistency_ratio <= 100.0
    assert doc.violation_count >= 0
