"""
preprocessing/sentencizer.py — STT 발화 → 완결 문장 재구성 (문장화)

[이수민 - 2026-06-15]
STT는 `<HH:MM:SS> id: text` 라인 파편이라 '문장'이 아니다. 항목 2(발화 완결성)·
3(언어 일관성)은 문장 단위 + 종결어미(EF/EC)가 필요하므로, 먼저 라인을 문장으로
재구성한다. 이 단계는 임베딩(KcBERT 등) 결정과 무관 — 두 경로 모두 '문장'을 입력으로 받는다.

파이프라인:
  1) segment_utterances()  — 타임스탬프 갭 > 임계값(기본 10초)을 발화 세그먼트 경계로 분리
                             (점심 갭 30분짜리 preprocessor.split_sessions 와 다른 granularity)
  2) 세그먼트 텍스트 merge  — 원본 라인 인덱스/타임스탬프를 char offset 으로 보존
  3) Kiwi 문장 분리         — kiwi.split_into_sents
  4) Kiwi EF/EC 판정        — 각 문장 마지막 실질 형태소가 EF=완결 / EC=불완결
  → list[Sentence] (타임스탬프·원본 인덱스·끊김 플래그·완결성)

결정 사항:
  - 형태소 분석기/문장 분리: Kiwi (kiwipiepy) — [이수민 2026-06-22] 전환.
    2026-06-11 팀 결정인 Mecab(python-mecab-ko)+kss 를 대체. 교체 전 호환 확인:
    EF/EC 태그셋 동일 사용, 정규화 표면형(ᆸ니다/ᆫ다/ᆯ까요 등)도 종결어미 끝 음절이
    보존돼 formality 의 endswith(요/죠·니다/니까·다) 매칭과 일치. 실데이터 3강의
    완결률·일관율 변화 미미함을 확인(스코어 밴드 영향 없음). ⚠️ 팀 결정 변경분이라 공유 필요.
  - 갭 임계값 30초: 루브릭 default 10초 → EDA로 상향(갭 p50 ~10초 = 발화 길이, 침묵 아님).
                    완결률이 30초부터 평탄화. 상세는 SEGMENT_GAP_SECONDS 주석.

주의:
  - Sentence 는 아직 LectureDocument 에 미편입 (다운스트림 호환 영향 검토 후 결정).
  - use_morph=False 면 문장 분리까지만 수행(완결성 미판정). Kiwi 는 필수.
"""
from __future__ import annotations

from app.analysis.schemas import Sentence, Utterance

# 발화 세그먼트 경계: 인접 발화 갭이 이 값을 넘으면 새 세그먼트.
# [이수민 - 2026-06-15] EDA로 10초 → 30초 상향. STT 타임스탬프가 발화 '시작점'이라
# 라인 간 갭은 침묵이 아니라 직전 발화 길이(15개 파일 갭 p50 ~10초)다. 10초로 끊으면
# 라인의 ~47%에서 분할돼 한 문장이 쪼개지고 완결률이 ~12%p 과소(59.7% vs 30초 72.2%).
# 완결률이 30초부터 평탄화(plateau) → 30초 채택. (루브릭 default 는 10초였음)
SEGMENT_GAP_SECONDS = 30

# 문장 끝 '실질 형태소' 탐색 시 건너뛸 기호 품사 (마침표/쉼표/줄임표/괄호 등) — Kiwi 태그셋.
# SL(외국어)·SH(한자)·SN(숫자)는 실질 내용이라 건너뛰지 않는다.
_SYMBOL_TAGS = frozenset({"SF", "SP", "SS", "SSO", "SSC", "SE", "SO", "SW"})

# 세그먼트 내 라인 merge 구분자 (공백 1칸)
_MERGE_SEP = " "

# 무거운 형태소 분석기는 모듈 1회만 로드 (지연 초기화)
_kiwi = None


# ── 형태소/문장 분리 백엔드 (Kiwi 단일 엔진) ──────────────────
def _get_kiwi():
    """kiwipiepy Kiwi 싱글턴. 형태소 분석·문장 분리 모두 담당."""
    global _kiwi
    if _kiwi is None:
        from kiwipiepy import Kiwi
        _kiwi = Kiwi()
    return _kiwi


def _split_korean_sentences(text: str) -> list[str]:
    """Kiwi 문장 분리 (kiwi.split_into_sents). 빈 문장 제거."""
    if not text.strip():
        return []
    kiwi = _get_kiwi()
    return [s.text for s in kiwi.split_into_sents(text) if s.text.strip()]


# ── 1) 갭 기반 발화 세그먼트 분리 ─────────────────────────────
def segment_utterances(
    utterances: list[Utterance],
    gap_threshold_seconds: int = SEGMENT_GAP_SECONDS,
) -> list[list[Utterance]]:
    """타임스탬프 갭 > 임계값 지점을 경계로 발화 세그먼트 분리.

    `preprocessor.split_sessions`(30분, 오전/오후) 와 granularity 가 다르다.
    이쪽은 호흡 단위(기본 10초) — 발화 끊김 탐지가 목적.
    """
    return [
        [u for _, u in seg]
        for seg in _segment_with_index(utterances, gap_threshold_seconds)
    ]


def _segment_with_index(
    utterances: list[Utterance],
    gap_threshold_seconds: int,
) -> list[list[tuple[int, Utterance]]]:
    """세그먼트 분리 + 원본 전역 인덱스 보존 (내부용)."""
    if not utterances:
        return []
    groups: list[list[tuple[int, Utterance]]] = [[(0, utterances[0])]]
    for idx in range(1, len(utterances)):
        prev, cur = utterances[idx - 1], utterances[idx]
        if cur.seconds_from_start - prev.seconds_from_start > gap_threshold_seconds:
            groups.append([(idx, cur)])
        else:
            groups[-1].append((idx, cur))
    return groups


# ── 2) 세그먼트 텍스트 merge + char offset 보존 ───────────────
def _merge_with_offsets(
    seg: list[tuple[int, Utterance]],
) -> tuple[str, list[tuple[int, int, int, Utterance]]]:
    """세그먼트 라인을 한 문자열로 merge + 각 라인의 char span 기록.

    반환: (merged_text, [(char_start, char_end, global_index, utterance), ...])
    문장 → 원본 라인(타임스탬프) 역추적에 사용.
    """
    parts: list[str] = []
    offsets: list[tuple[int, int, int, Utterance]] = []
    cursor = 0
    for k, (gidx, utt) in enumerate(seg):
        if k > 0:
            cursor += len(_MERGE_SEP)
        text = utt.text
        offsets.append((cursor, cursor + len(text), gidx, utt))
        parts.append(text)
        cursor += len(text)
    return _MERGE_SEP.join(parts), offsets


# ── 4) 문장 마지막 형태소 EF/EC 판정 ──────────────────────────
def _classify_ending(text: str, kiwi) -> tuple[str | None, str | None, bool | None]:
    """문장 마지막 '실질' 형태소의 (표면형, 품사, 완결여부).

    완결 = 마지막 실질 형태소가 EF(종결어미). EC(연결어미) 등은 불완결.
    기호(SF/SP/SS/...)는 건너뛴다. 형태소 없음 → (None, None, None).
    Kiwi 표면형은 정규화형(ᆸ니다/ᆫ다 등)이나 종결어미 끝 음절이 보존돼
    formality 의 endswith 매칭과 호환된다. 태그는 단일이라 '+' 분해는 안전 차원.
    """
    for tok in reversed(kiwi.tokenize(text)):
        sub_tags = tok.tag.split("+")
        if all(t in _SYMBOL_TAGS for t in sub_tags):
            continue  # 마침표 등 기호는 건너뜀
        return tok.form, tok.tag, ("EF" in sub_tags)
    return None, None, None


# ── 통합: Utterance 리스트 → Sentence 리스트 ──────────────────
def build_sentences(
    utterances: list[Utterance],
    gap_threshold_seconds: int = SEGMENT_GAP_SECONDS,
    use_morph: bool = True,
) -> list[Sentence]:
    """STT Utterance 리스트 → 재구성된 Sentence 리스트.

    1) 갭 기반 세그먼트 분리 2) merge 3) Kiwi 문장 분리 4) (옵션) Kiwi EF/EC 완결성.
    원본 라인 인덱스·타임스탬프·끊김 플래그를 보존한다.
    """
    segments = _segment_with_index(utterances, gap_threshold_seconds)
    kiwi = _get_kiwi() if use_morph else None
    sentences: list[Sentence] = []

    for seg_idx, seg in enumerate(segments):
        is_last_segment = seg_idx == len(segments) - 1
        merged, offsets = _merge_with_offsets(seg)
        raw_sents = _split_korean_sentences(merged)

        cursor = 0
        for s_idx, s in enumerate(raw_sents):
            pos = merged.find(s, cursor)
            if pos < 0:                      # kss 가 공백 정규화한 경우 폴백
                pos = cursor
            span_start, span_end = pos, pos + len(s)
            cursor = span_end

            covered = [
                (a, b, gidx, utt)
                for (a, b, gidx, utt) in offsets
                if a < span_end and b > span_start
            ]
            if not covered:                  # 매핑 실패 → 가장 가까운 라인
                covered = [min(offsets, key=lambda o: abs(o[0] - span_start))]

            first_utt, last_utt = covered[0][3], covered[-1][3]
            # 세그먼트의 마지막 문장이고 뒤에 또 세그먼트가 있으면 = 갭 직전 = 끊김 후보
            ends_with_gap = (s_idx == len(raw_sents) - 1) and not is_last_segment

            if kiwi is not None:
                ending_morph, ending_tag, is_complete = _classify_ending(s, kiwi)
            else:
                ending_morph, ending_tag, is_complete = None, None, None

            sentences.append(
                Sentence(
                    text=s,
                    segment_index=seg_idx,
                    start_timestamp=first_utt.timestamp,
                    end_timestamp=last_utt.timestamp,
                    start_seconds=first_utt.seconds_from_start,
                    end_seconds=last_utt.seconds_from_start,
                    utterance_indices=[gidx for (_, _, gidx, _) in covered],
                    ends_with_gap=ends_with_gap,
                    ending_morph=ending_morph,
                    ending_tag=ending_tag,
                    is_complete=is_complete,
                )
            )
    return sentences


# ── 항목 2 지표 씨앗: 완결 문장 비율 ──────────────────────────
def completeness_rate(sentences: list[Sentence]) -> float:
    """완결 문장 비율(%) — 항목 2(발화 완결성) 지표의 씨앗.

    완결 = is_complete True. is_complete None(미판정)은 분모에서 제외.
    """
    judged = [s for s in sentences if s.is_complete is not None]
    if not judged:
        return 0.0
    complete = sum(1 for s in judged if s.is_complete)
    return complete / len(judged) * 100


# ── 항목 2 채점 (잠정 밴드) ────────────────────────────────────
# [이수민 - 2026-06-22] 잠정 밴드. [2026-06-22 재보정] Mecab→Kiwi 전환 반영.
# 형태소 분석기를 Kiwi 로 바꾸자 완결률이 69~72%(Mecab) → 96.5%(Kiwi, σ0.4)로 상승.
#   원인: Mecab 이 구어 종결어미(게요/냐/죠/어요/구나/야 등)를 EC 로 오태깅해 완결을
#   과소집계하던 것을 Kiwi 가 바로잡음(15강의 26%가 Mecab非EF→Kiwi EF, 검수 결과 Kiwi 정답).
# 근거 부재 주의: 출판 컷오프 없음. 아래는 Kiwi 실측(평균 96.5%)을 '거의 완전(5)'에 두고
#   trailing-off(EC 종결)가 늘수록 감점하는 **내부 잠정 기준**(출판 근거 아님).
#   주의: 정확 태깅 하에선 유창한 강사의 완결률이 포화(≈96%)라 변별력은 약함 — 절대 등급 용도.
# 적용 범위: 단일 강사(팀 안내 2026-06-22). 컷오프 확정 시 docs/scoring-bands.md 갱신 + 팀 협의.
def completeness_score(rate: float) -> int:
    """completeness_rate(%) → 1~5 (잠정 밴드, Kiwi 분포 기준). 상세는 위 주석/문서."""
    if rate >= 93:
        return 5
    if rate >= 87:
        return 4
    if rate >= 78:
        return 3
    if rate >= 68:
        return 2
    return 1
