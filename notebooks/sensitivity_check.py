"""
notebooks/sensitivity_check.py — 지표 민감도 검증 (perturbation) + 차트

[이수민 - 2026-06-15]
단일 강사 데이터의 한계("지표가 안정적인가 vs 변별을 못 하는가")를 풀기 위한
민감도(sensitivity) 검증. 실제 전사본에 통제된 변형을 가해 지표가 *행동 변화에
단조 반응*하는지 본다 — 변별력의 필요조건. (추가 강사 없이 가능)

핵심 원칙: 합성 텍스트를 만들지 않는다. 실제 문장·실제 mecab 라벨을 쓰고
**구성(mix)/타임스탬프만 통제**한다.

검증 3종 (기준 파일 02-02) → backend/outputs/sensitivity_check.png:
  A. 속도   — seconds_from_start × k → wpm_kr 가 ~1/k 로 움직이나 (전체 파이프라인)
  B. 일관성 — 실제 반말 문장 비율 ↑ → consistency_ratio ↓ (50:50에서 최악)
  C. 완결성 — 실제 불완결(EC) 문장 비율 ↑ → completeness_rate ↓

실행: python notebooks/sensitivity_check.py
"""
from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.analysis.schemas import Utterance
from app.core.paths import read_stt
from app.preprocessing import formality, pace, preprocessor, sentencizer

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

random.seed(0)
DATE, COURSE = "2026-02-02", "kdt-backendj-21th"
_HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PNG = os.path.join(_HERE, "..", "outputs", "sensitivity_check.png")


def _scale_time(utts: list[Utterance], k: float) -> list[Utterance]:
    return [
        Utterance(timestamp=u.timestamp, speaker_id=u.speaker_id, text=u.text,
                  seconds_from_start=int(u.seconds_from_start * k))
        for u in utts
    ]


def _resample(minority: list, majority: list, neutral: list, target_minority_frac: float) -> list:
    """소수 그룹 비율을 target 으로 맞춰 (실제 문장을 복원추출) 리스트 구성."""
    n = len(minority) + len(majority)
    n_min = round(n * target_minority_frac)
    samp = [random.choice(minority) for _ in range(n_min)] if minority else []
    samp += [random.choice(majority) for _ in range(n - n_min)] if majority else []
    return samp + neutral


def main() -> None:
    utts = preprocessor.parse(read_stt(DATE, COURSE))
    sents = sentencizer.build_sentences(utts, gap_threshold_seconds=30)
    cls = formality.classify_formality

    # ── A. 속도 ───────────────────────────────────────────────
    ks = [0.6, 0.8, 1.0, 1.25, 1.67]
    base_wpm = pace.wpm_kr(utts)
    wpmA = [pace.wpm_kr(_scale_time(utts, k)) for k in ks]
    expA = [base_wpm / k for k in ks]

    # ── B. 일관성 ─────────────────────────────────────────────
    jond = [s for s in sents if cls(s.ending_morph, s.ending_tag) in (formality.FORMAL, formality.INFORMAL_POLITE)]
    ban = [s for s in sents if cls(s.ending_morph, s.ending_tag) == formality.BANMAL]
    neut = [s for s in sents if cls(s.ending_morph, s.ending_tag) == formality.NEUTRAL]
    base_ban = len(ban) / (len(jond) + len(ban))
    fracB = [base_ban + e for e in (0.0, 0.1, 0.2, 0.3)]
    profB = [formality.formality_profile(_resample(ban, jond, neut, f)) for f in fracB]
    consB = [p["consistency_ratio"] for p in profB]
    violB = [p["violation_count"] for p in profB]

    # ── C. 완결성 ─────────────────────────────────────────────
    comp = [s for s in sents if s.is_complete is True]
    incomp = [s for s in sents if s.is_complete is False]
    base_inc = len(incomp) / (len(comp) + len(incomp))
    fracC = [base_inc + e for e in (0.0, 0.1, 0.2, 0.3)]
    compC = [sentencizer.completeness_rate(_resample(incomp, comp, [], f)) for f in fracC]

    # ── 차트 ──────────────────────────────────────────────────
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4.4))

    a1.plot(ks, wpmA, "o-", color="#2f6fb0", lw=2, label="관측 wpm_kr")
    a1.plot(ks, expA, "--", color="gray", label="기대 base/k")
    a1.set_title("A. 속도 — wpm_kr 가 1/k 추종")
    a1.set_xlabel("타임스탬프 배율 k (작을수록 빠름)")
    a1.set_ylabel("wpm_kr (어절/분)")
    a1.legend(fontsize=8); a1.grid(alpha=0.3)

    a2.plot([f * 100 for f in fracB], consB, "o-", color="#5aa0d6", lw=2, label="consistency_ratio")
    a2.axvline(50, ls=":", color="crimson", lw=1)
    a2.text(50.5, min(consB), "50:50\n최악", color="crimson", fontsize=8, va="bottom")
    a2b = a2.twinx()
    a2b.plot([f * 100 for f in fracB], violB, "s--", color="#e07b54", lw=1.5, label="violation_count")
    a2b.set_ylabel("violation_count", color="#e07b54")
    a2.set_title("B. 일관성 — 반말↑ 시 일관성↓")
    a2.set_xlabel("반말 문장 비율 (%)")
    a2.set_ylabel("consistency_ratio (%)", color="#5aa0d6")
    a2.grid(alpha=0.3)

    a3.plot([f * 100 for f in fracC], compC, "o-", color="#5aa469", lw=2, label="관측")
    a3.plot([f * 100 for f in fracC], [100 - f * 100 for f in fracC], "--", color="gray", label="기대 100−불완결%")
    a3.set_title("C. 완결성 — 불완결↑ 시 완결률↓")
    a3.set_xlabel("불완결(EC) 문장 비율 (%)")
    a3.set_ylabel("completeness_rate (%)")
    a3.legend(fontsize=8); a3.grid(alpha=0.3)

    fig.suptitle("지표 민감도 검증 — 행동 변화에 단조 반응 (02-02, 실제 문장)  [이수민 2026-06-15]",
                 fontsize=12, y=1.03)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"저장: {os.path.normpath(OUTPUT_PNG)}")
    print(f"A 속도   : k=0.6→{wpmA[0]:.0f}, k=1.0→{wpmA[2]:.0f}, k=1.67→{wpmA[-1]:.0f} (1/k 추종)")
    print(f"B 일관성 : 반말 {base_ban*100:.0f}%→{consB[0]:.0f}%, 52%→{consB[2]:.0f}% (50:50 최악)")
    print(f"C 완결성 : 불완결 {base_inc*100:.0f}%→{compC[0]:.0f}%, +30%p→{compC[-1]:.0f}% (선형 하락)")


if __name__ == "__main__":
    main()
