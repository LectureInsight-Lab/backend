"""
tests/test_sentencizer.py — 문장화(sentencizer) 회귀 테스트

[이수민 - 2026-06-15]
검증:
  - segment_utterances: 갭 > 임계값에서 발화 세그먼트 분리
  - build_sentences: kss 문장 분리 + Mecab EF/EC 완결성 판정
      · 'VV+EF' 종결 → is_complete True
      · 'EC' 종결    → is_complete False
  - ends_with_gap: 세그먼트 끝(뒤 갭) 문장에 끊김 플래그
  - completeness_rate: 0~100 범위

Mecab(python-mecab-ko)·kss 미설치 환경에서는 자동 skip.
"""
from __future__ import annotations

import pytest

from app.analysis.schemas import Utterance
from app.preprocessing import sentencizer

# 형태소/문장 분리 백엔드 없으면 전체 skip
pytest.importorskip("kss")
pytest.importorskip("mecab")


def _utt(ts: str, secs: int, text: str) -> Utterance:
    return Utterance(timestamp=ts, speaker_id="a", text=text, seconds_from_start=secs)


def test_segment_splits_on_gap():
    utts = [
        _utt("09:00:00", 0, "안녕하세요"),
        _utt("09:00:05", 5, "오늘 시작합니다"),
        _utt("09:00:40", 40, "다음 내용입니다"),   # 35초 갭 → 새 세그먼트
    ]
    segs = sentencizer.segment_utterances(utts, gap_threshold_seconds=10)
    assert len(segs) == 2
    assert [len(s) for s in segs] == [2, 1]


def test_complete_ending_is_ef():
    utts = [_utt("09:00:00", 0, "오늘은 자바를 배웁니다")]
    sents = sentencizer.build_sentences(utts, use_mecab=True)
    assert len(sents) == 1
    assert sents[0].is_complete is True
    assert "EF" in sents[0].ending_tag


def test_incomplete_ending_is_ec():
    utts = [_utt("09:00:00", 0, "그래서 이게 중요하고")]
    sents = sentencizer.build_sentences(utts, use_mecab=True)
    assert len(sents) == 1
    assert sents[0].is_complete is False
    assert "EC" in sents[0].ending_tag


def test_ends_with_gap_flag():
    utts = [
        _utt("09:00:00", 0, "첫 문장입니다"),
        _utt("09:00:30", 30, "둘째 세그먼트 문장입니다"),   # 30초 갭
    ]
    sents = sentencizer.build_sentences(utts, gap_threshold_seconds=10)
    assert sents[0].ends_with_gap is True       # 첫 세그먼트 끝 = 갭 직전
    assert sents[-1].ends_with_gap is False      # 마지막 세그먼트


def test_timestamps_preserved_across_merge():
    # 한 문장이 두 라인에 걸쳐 있으면 start/end 타임스탬프가 양 끝을 가리킨다
    utts = [
        _utt("09:00:00", 0, "자바는"),
        _utt("09:00:03", 3, "객체지향 언어입니다"),
    ]
    sents = sentencizer.build_sentences(utts)
    merged = sents[0]
    assert merged.start_seconds == 0
    assert merged.end_seconds == 3
    assert merged.utterance_indices == [0, 1]


def test_completeness_rate_range():
    utts = [
        _utt("09:00:00", 0, "자바는 객체지향 언어입니다"),
        _utt("09:00:04", 4, "클래스를 정의하고"),
    ]
    rate = sentencizer.completeness_rate(sentencizer.build_sentences(utts))
    assert 0.0 <= rate <= 100.0


def test_use_mecab_false_skips_completeness():
    utts = [_utt("09:00:00", 0, "오늘은 자바를 배웁니다")]
    sents = sentencizer.build_sentences(utts, use_mecab=False)
    assert sents[0].is_complete is None
    assert sents[0].ending_tag is None
