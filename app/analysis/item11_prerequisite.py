"""
항목 3-3: 선행 개념 확인 (prerequisite_coverage)

입력: raw STT .txt 파일 경로
  형식: <HH:MM:SS> speaker_id: text

절차:
  1. 2분 단위 청크 분할 (12시간 시계 delta 보정)
  2. 인접 청크 간 KR-SBERT 코사인 유사도 계산
  3. 하위 25% 백분위수 미만 = 주제 전환점
  4-1. 전환점 직전·직후 청크에서 regex 선행 확인 표현 탐지
  4-2. semantic search 보완 (SEMANTIC_THRESHOLD 이상이면 확인으로 인정)
  5. prerequisite_coverage = 확인 전환점 / 전체 전환점 × 100

채점: 5≥85% / 4≥70% / 3≥50% / 2≥30% / 1<30%

출력:
  {"evidence": list[object] | null, "final_score": int | "N/A"}
  mode=""      → evidence: null
  mode="debug" → evidence: 전환점별 탐지 결과 리스트
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from loguru import logger

from app.analysis.embedder import cosine_sim, embed

CHUNK_MINUTES      = 2
SHIFT_PERCENTILE   = 25
SEMANTIC_THRESHOLD = 0.50

_LINE_RE = re.compile(r"^<(\d{2}:\d{2}:\d{2})>\s+\S+:\s*(.*)$")

_PREREQ_RE = re.compile(
    # 시간 참조 (아까, 방금 전에, 예전에)
    r"아까.{0,40}(있었|봤|했|봤는데|했는데|얘기|설명|말씀|봤잖|했잖|나왔|나온|리스트|메소드|거죠|하셨|하셨죠)"
    r"|아까도\b"
    r"|방금 전에"
    r"|예전에.{0,15}(이거|그거|이걸|봤|했|썼|썼죠|써봤|해봤)"
    r"|써봤죠|해봤죠|봤죠\b"

    # 명시적 시간 지시어
    r"|지난.{0,5}(번|시간|수업)"
    r"|저번.{0,5}(에|시간|수업)"
    r"|어제.{0,15}(했던|봤던|배운|다뤘던|한|설명)"
    r"|지난번에|전에.{0,8}(했던|봤던|배운|다뤘|본)"
    r"|맨 처음에.{0,25}(했던|봤던|배운|했잖|봤잖)"

    # 과거 경험 회상 (했잖아요, 봤잖아요)
    r"|해봤잖.{0,3}(아요|아|죠)|했잖.{0,3}(아요|아|죠)|봤잖.{0,3}(아요|아|죠)"
    r"|배웠잖|출력해봤잖|따라해봤잖"
    r"|해봤으니까|만졌으니까|해보셨으니까|배웠으니까|봤으니까"
    r"|봤었죠|했었죠|배웠었죠|봤었는데|했었는데"
    r"|하라고 했(어|잖|던)|보라고 했(어|잖|던)"

    # 명시적 회상 요청
    r"|기억하시(죠|나요|나|는)|기억나시(죠|나요|나|는)"
    r"|기억.{0,10}(하고|하죠|날|납니다)"
    r"|기억해요|기억해봐|기억해보세요"

    # 앞서/이전에
    r"|앞서.{0,15}(봤던|했던|설명|배운|다룬|언급|말씀|봤는데)"
    r"|이전에.{0,15}(배운|했던|봤던|설명|다룬|말씀|봤는데)"
    r"|이미.{0,8}(배운|했던|봤던|아시는|알고)"
    r"|앞서 설명|이전에 말씀"

    # 이어서/연결
    r"|이어서|연결해서|이어지는"
    r"|오늘은.{0,15}(이어서|이전|그래서)"
)

_PREREQ_REFS = [
    "이전에 배운 내용을 기반으로 오늘 내용을 시작하겠습니다",
    "앞서 다뤘던 개념과 연결해서 설명드리겠습니다",
    "저번에 배운 것 기억나시죠? 그걸 바탕으로 진행할게요",
    "먼저 이전 내용을 짚고 넘어가겠습니다",
    "오늘 내용은 지난 시간 내용과 이어집니다",
]


def score_prerequisite(txt_path: str | Path, mode: str = "") -> dict:
    """
    Parameters
    ----------
    txt_path : str | Path
        raw STT .txt 파일 경로.
    mode : str
        "" (default) → evidence: null / "debug" → evidence: 전환점별 탐지 결과
    """
    chunks = _build_2min_chunks(Path(txt_path))

    if len(chunks) < 4:
        logger.warning(f"[item11] 청크 {len(chunks)}개 — 분석 불가")
        return _build(score="N/A", evidence_items=[], mode=mode)

    logger.info(f"[item11] 2분 청크 {len(chunks)}개 생성")

    # Step 2: 인접 청크 유사도
    embeddings = embed(chunks)
    sims = [cosine_sim(embeddings[i], embeddings[i + 1]) for i in range(len(embeddings) - 1)]

    # Step 3: 전환점 탐지
    threshold    = float(np.percentile(sims, SHIFT_PERCENTILE))
    shift_indices = [i for i, s in enumerate(sims) if s < threshold]

    if not shift_indices:
        return _build(score=5, evidence_items=[], mode=mode)

    # Step 4-2: 레퍼런스 임베딩
    ref_embs = embed(_PREREQ_REFS)

    confirmed     = 0
    evidence_items: list[dict] = []

    for idx in shift_indices:
        check_pairs = [("직전", chunks[idx])]
        if idx + 1 < len(chunks):
            check_pairs.append(("직후", chunks[idx + 1]))

        found_hit = None
        for pos, text in check_pairs:
            hit = _PREREQ_RE.search(text)
            if hit:
                found_hit = f"regex({pos}, '{hit.group()[:25]}')"
                break
            sem = float(max(cosine_sim(embed([text])[0], r) for r in ref_embs))
            if sem >= SEMANTIC_THRESHOLD:
                found_hit = f"semantic({pos}, {sem:.2f})"
                break

        chunk_min = idx * CHUNK_MINUTES
        if found_hit:
            confirmed += 1

        evidence_items.append({
            "chunk_min": chunk_min,
            "confirmed": found_hit is not None,
            "method": found_hit,
            "text_preview": chunks[idx][:80],
        })

    total    = len(shift_indices)
    coverage = confirmed / total * 100

    logger.info(f"[item11] 전환점 {total}개, 확인 {confirmed}개 → coverage={coverage:.1f}%")

    score = (
        5 if coverage >= 85 else
        4 if coverage >= 70 else
        3 if coverage >= 50 else
        2 if coverage >= 30 else
        1
    )

    return _build(score=score, evidence_items=evidence_items, mode=mode)


def _ts_to_sec(ts: str) -> int:
    h, m, s = map(int, ts.split(":"))
    return h * 3600 + m * 60 + s


def _build_2min_chunks(txt_path: Path, chunk_min: int = CHUNK_MINUTES) -> list[str]:
    """
    raw STT 파일을 읽어 2분 단위 텍스트 청크 리스트를 반환한다.
    12시간 시계 문제(오전 09:xx → 오후 01:xx)는 인접 라인 delta로 보정한다.
    """
    lines = txt_path.read_text(encoding="utf-8").splitlines()

    parsed: list[tuple[str, str]] = []
    for line in lines:
        m = _LINE_RE.match(line.strip())
        if m:
            parsed.append((m.group(1), m.group(2).strip()))

    if not parsed:
        return []

    chunk_sec   = chunk_min * 60
    running_sec = 0
    prev_abs    = _ts_to_sec(parsed[0][0])
    buckets: dict[int, list[str]] = {}

    for ts, text in parsed:
        abs_sec = _ts_to_sec(ts)
        delta   = abs_sec - prev_abs

        if delta < -3600:
            delta += 12 * 3600
        elif delta < 0:
            delta = 0

        running_sec += delta
        prev_abs     = abs_sec

        bucket = running_sec // chunk_sec
        buckets.setdefault(bucket, []).append(text)

    return [" ".join(texts) for texts in [buckets[k] for k in sorted(buckets)]]


def _build(score: int | str, evidence_items: list[dict], mode: str) -> dict:
    return {
        "evidence": evidence_items if mode == "debug" else None,
        "final_score": score,
    }
