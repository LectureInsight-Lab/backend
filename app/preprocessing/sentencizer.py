"""
preprocessing/sentencizer.py — STT 발화 → 완결 문장 재구성 (문장화)

[이수민 - 2026-06-15]
STT는 `<HH:MM:SS> id: text` 라인 파편이라 '문장'이 아니다. 항목 2(발화 완결성)·
3(언어 일관성)은 문장 단위 + 종결어미(EF/EC)가 필요하므로, 먼저 라인을 문장으로
재구성한다. 이 단계는 Mecab/KcBERT 결정과 무관 — 두 경로 모두 '문장'을 입력으로 받는다.

파이프라인:
  1) segment_utterances()  — 타임스탬프 갭 > 임계값(기본 10초)을 발화 세그먼트 경계로 분리
                             (점심 갭 30분짜리 preprocessor.split_sessions 와 다른 granularity)
  2) 세그먼트 텍스트 merge  — 원본 라인 인덱스/타임스탬프를 char offset 으로 보존
  3) kss 문장 분리          — mecab 백엔드 자동 사용
  4) Mecab EF/EC 판정       — 각 문장 마지막 실질 형태소가 EF=완결 / EC=불완결
  → list[Sentence] (타임스탬프·원본 인덱스·끊김 플래그·완결성)

결정 사항:
  - 형태소 분석기: Mecab (python-mecab-ko, mecab-ko-dic 번들) — 2026-06-11 팀 결정
  - 문장 분리: kss (Korean Sentence Splitter)
  - 갭 임계값 30초: 루브릭 default 10초 → EDA로 상향(갭 p50 ~10초 = 발화 길이, 침묵 아님).
                    완결률이 30초부터 평탄화. 상세는 SEGMENT_GAP_SECONDS 주석.

주의:
  - Sentence 는 아직 LectureDocument 에 미편입 (다운스트림 호환 영향 검토 후 결정).
  - use_mecab=False 면 문장 분리까지만 수행(완결성 미판정). kss 는 필수.
"""
from __future__ import annotations

from app.analysis.schemas import Sentence, Utterance

# 발화 세그먼트 경계: 인접 발화 갭이 이 값을 넘으면 새 세그먼트.
# [이수민 - 2026-06-15] EDA로 10초 → 30초 상향. STT 타임스탬프가 발화 '시작점'이라
# 라인 간 갭은 침묵이 아니라 직전 발화 길이(15개 파일 갭 p50 ~10초)다. 10초로 끊으면
# 라인의 ~47%에서 분할돼 한 문장이 쪼개지고 완결률이 ~12%p 과소(59.7% vs 30초 72.2%).
# 완결률이 30초부터 평탄화(plateau) → 30초 채택. (루브릭 default 는 10초였음)
SEGMENT_GAP_SECONDS = 30

# 문장 끝 '실질 형태소' 탐색 시 건너뛸 기호 품사 (마침표/줄임표/괄호 등)
_SYMBOL_TAGS = frozenset({"SF", "SE", "SSO", "SSC", "SC", "SY"})

# 세그먼트 내 라인 merge 구분자 (공백 1칸)
_MERGE_SEP = " "

# 무거운 형태소 분석기는 모듈 1회만 로드 (지연 초기화)
_mecab = None


# ── 형태소/문장 분리 백엔드 ───────────────────────────────────
def _get_mecab():
    """python-mecab-ko MeCab 싱글턴 (mecab-ko-dic 번들)."""
    global _mecab
    if _mecab is None:
        from mecab import MeCab
        _mecab = MeCab()
    return _mecab


def _split_korean_sentences(text: str) -> list[str]:
    """kss 문장 분리 (mecab 백엔드 자동). 빈 문장 제거."""
    import kss
    if not text.strip():
        return []
    return [s for s in kss.split_sentences(text) if s.strip()]


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
def _classify_ending(text: str, mecab) -> tuple[str | None, str | None, bool | None]:
    """문장 마지막 '실질' 형태소의 (표면형, 품사, 완결여부).

    완결 = 마지막 실질 형태소가 EF(종결어미). EC(연결어미) 등은 불완결.
    기호(SF/SE/...)는 건너뛴다. 형태소 없음 → (None, None, None).
    품사는 복합 태그('VV+EF')일 수 있어 '+' 분해 후 EF 포함 여부로 판정.
    """
    for surface, tag in reversed(mecab.pos(text)):
        sub_tags = tag.split("+")
        if all(t in _SYMBOL_TAGS for t in sub_tags):
            continue  # 마침표 등 기호는 건너뜀
        return surface, tag, ("EF" in sub_tags)
    return None, None, None


# ── 통합: Utterance 리스트 → Sentence 리스트 ──────────────────
def build_sentences(
    utterances: list[Utterance],
    gap_threshold_seconds: int = SEGMENT_GAP_SECONDS,
    use_mecab: bool = True,
) -> list[Sentence]:
    """STT Utterance 리스트 → 재구성된 Sentence 리스트.

    1) 갭 기반 세그먼트 분리 2) merge 3) kss 문장 분리 4) (옵션) Mecab EF/EC 완결성.
    원본 라인 인덱스·타임스탬프·끊김 플래그를 보존한다.
    """
    segments = _segment_with_index(utterances, gap_threshold_seconds)
    mecab = _get_mecab() if use_mecab else None
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

            if mecab is not None:
                ending_morph, ending_tag, is_complete = _classify_ending(s, mecab)
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
# [이수민 - 2026-06-22] 잠정 score band 추가.
# 근거 부재 주의: 한국어 구어 강의 '완결률'의 출판 컷오프 기준치는 없음. 아래 밴드는
#   ① 실측(15강의 완결률 69~72%, 편차 ±2.0)을 '정상~양호(4)' 위치에 두고
#   ② 구어 특성상 100% 완결은 비자연(자기수정·중단 정상)이라 천장을 낮춰 잡은
#   **내부 잠정 기준**이다. 출판 근거 기반 아님.
# 적용 범위: 팀 안내(2026-06-22) — 현재는 단일 강사만 대상. 강사 간 변별 보정 불요.
#   단일 강사라 15강의 점수는 거의 일정(정상). 변별이 아니라 절대 등급 + 주차 trend 용도.
# 컷오프 확정 시 docs/scoring-bands.md 갱신 + 팀 협의.
def completeness_score(rate: float) -> int:
    """completeness_rate(%) → 1~5 (잠정 밴드). 상세 근거는 위 주석/문서 참고."""
    if rate >= 80:
        return 5
    if rate >= 70:
        return 4
    if rate >= 60:
        return 3
    if rate >= 50:
        return 2
    return 1
