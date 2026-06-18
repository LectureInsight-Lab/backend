"""
notebooks/eda_item02_completeness.py — 항목 2(발화 완결성) EDA + 차트 생성

[이수민 - 2026-06-15]
문장화(preprocessing/sentencizer)를 실제 강의 STT 15개에 적용해 두 가지를 그린다:
  (좌) 갭 임계값 스윕(02-02) — 완결률이 30초부터 평탄화(plateau)함을 확인
  (우) 15개 강의 완결률 분포(임계값 30초) 히스토그램
→ backend/outputs/eda_item02_completeness.png 저장.

핵심 결론(2026-06-15 보고서):
  - STT 타임스탬프 = 발화 시작점 → 라인 간 갭 p50 ~10초는 침묵이 아닌 '발화 길이'
  - 갭 임계값 10초 → 30초 상향 (완결률 +12.5%p, 30초부터 plateau)
  - 단일 강사 15강의 완결률 72.2% ± 2.0 (강사 간 변별력은 데이터 부족으로 미검증)

실행 (backend 디렉터리에서):
  python notebooks/eda_item02_completeness.py
필요 데이터: configs/paths.yaml 의 stt_dir (data/강의 스크립트/).
"""
from __future__ import annotations

import glob
import os
import sys

# 스크립트를 직접 실행(python notebooks/eda_item02_completeness.py)해도 app 패키지를
# 찾도록 backend 디렉터리를 import 경로에 추가.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")  # 헤드리스 저장
import matplotlib.pyplot as plt
import numpy as np

from app.core.paths import load_paths, read_stt
from app.preprocessing import preprocessor, sentencizer

# 한글 폰트 (macOS 기본)
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

SWEEP_THRESHOLDS = [10, 15, 20, 30, 45, 60, 120]
SWEEP_FILE = ("2026-02-02", "kdt-backendj-21th")
CHOSEN_THRESHOLD = 30

_HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PNG = os.path.join(_HERE, "..", "outputs", "eda_item02_completeness.png")


def _iter_files():
    """stt_dir 의 모든 STT 파일을 (date, course_id) 로 순회."""
    stt_dir = str(load_paths().data.stt_dir)
    for path in sorted(glob.glob(os.path.join(stt_dir, "*.txt"))):
        base = os.path.basename(path)
        date, rest = base.split("_", 1)
        yield date, rest.rsplit(".", 1)[0]


def sweep_completeness(date: str, course: str) -> list[float]:
    """단일 파일의 갭 임계값별 완결률 (plateau 확인용)."""
    utts = preprocessor.parse(read_stt(date, course))
    return [
        sentencizer.completeness_rate(
            sentencizer.build_sentences(utts, gap_threshold_seconds=t)
        )
        for t in SWEEP_THRESHOLDS
    ]


def per_file_completeness(threshold: int) -> dict[str, float]:
    """모든 파일의 완결률 (분포용)."""
    out: dict[str, float] = {}
    for date, course in _iter_files():
        utts = preprocessor.parse(read_stt(date, course))
        sents = sentencizer.build_sentences(utts, gap_threshold_seconds=threshold)
        out[date] = sentencizer.completeness_rate(sents)
    return out


def main() -> None:
    print("갭 임계값 스윕 (02-02) ...")
    sweep = sweep_completeness(*SWEEP_FILE)
    print("15개 파일 완결률 집계 (임계값 30초) ...")
    comp = per_file_completeness(CHOSEN_THRESHOLD)
    vals = np.array(list(comp.values()))
    mean, std = vals.mean(), vals.std()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    # (좌) 임계값 스윕
    ax1.plot(SWEEP_THRESHOLDS, sweep, marker="o", color="#2f6fb0", lw=2)
    ax1.axvline(CHOSEN_THRESHOLD, ls="--", color="crimson", lw=1.5)
    ax1.annotate(
        f"채택 {CHOSEN_THRESHOLD}초\n(plateau 시작)",
        xy=(CHOSEN_THRESHOLD, sweep[SWEEP_THRESHOLDS.index(CHOSEN_THRESHOLD)]),
        xytext=(CHOSEN_THRESHOLD + 18, sweep[0] + 2),
        color="crimson", fontsize=9,
        arrowprops=dict(arrowstyle="->", color="crimson"),
    )
    ax1.set_title("갭 임계값별 완결률 (02-02, 1484라인)")
    ax1.set_xlabel("갭 임계값 (초)")
    ax1.set_ylabel("완결률 (%)")
    ax1.grid(alpha=0.3)

    # (우) 15개 강의 완결률 분포
    ax2.hist(vals, bins=np.arange(68, 78, 1), color="#5aa469", edgecolor="white")
    ax2.axvline(mean, color="crimson", ls="--", lw=1.5)
    ax2.text(mean + 0.15, ax2.get_ylim()[1] * 0.9,
             f"평균 {mean:.1f}%\n±{std:.1f}", color="crimson", fontsize=9)
    ax2.set_title("15개 강의 완결률 분포 (임계값 30초)")
    ax2.set_xlabel("완결률 (%)")
    ax2.set_ylabel("강의 수")
    ax2.grid(alpha=0.3, axis="y")

    fig.suptitle("항목 2 발화 완결성 — EDA  (2026-06-15, 이수민)", fontsize=13, y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")

    print(f"\n저장: {os.path.normpath(OUTPUT_PNG)}")
    print(f"완결률(30초): 평균 {mean:.1f}  범위 {vals.min():.1f}~{vals.max():.1f}  std {std:.1f}")


if __name__ == "__main__":
    main()
