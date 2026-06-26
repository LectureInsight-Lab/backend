"""
notebooks/eda_item03_consistency.py — 항목 3(언어 일관성) EDA + 차트 생성

[이수민 - 2026-06-15]
sentencizer + formality 를 실제 강의 STT 15개에 적용해:
  (좌) consistency_ratio 분포 히스토그램 (지배 말투 비율)
  (우) 강의별 말투 구성 누적막대 (격식/비격식/반말/중립)
→ backend/outputs/eda_item03_consistency.png 저장.

핵심 결론(2026-06-15):
  - 15개 전 파일 지배 말투 = 존댓말. consistency_ratio 평균 72.3% ± 4.6 (64.8~79.3)
  - 강사가 존댓말(해요체 중심) ~72% + 반말(야/어/잖아) ~28% 혼용
  - '-다'(한다체 평서)는 중립으로 분리 — 반말로 세면 일관성 폭락(formality.py 참고)
  - violation_count 평균 283 (강의당 비지배 말투 문장 수)
  - 채점 밴드: 관측 분포(72%)가 루브릭 <75=1점 구간 → 재보정 필요(항목 2와 동일)

실행 (backend 디렉터리에서):
  python notebooks/eda_item03_consistency.py
"""
from __future__ import annotations

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from app.core.paths import load_paths, read_stt
from app.preprocessing import item03_consistency as formality, preprocessor, sentencizer

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

CHOSEN_THRESHOLD = 30
_HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PNG = os.path.join(_HERE, "..", "outputs", "eda_item03_consistency.png")


def collect():
    """파일별 (date, formality_profile) 리스트."""
    out = []
    for path in sorted(glob.glob(os.path.join(str(load_paths().data.stt_dir), "*.txt"))):
        base = os.path.basename(path)
        date, rest = base.split("_", 1)
        utts = preprocessor.parse(read_stt(date, rest.rsplit(".", 1)[0]))
        sents = sentencizer.build_sentences(utts, gap_threshold_seconds=CHOSEN_THRESHOLD)
        out.append((date[5:], formality.formality_profile(sents)))
    return out


def main() -> None:
    print("15개 파일 말투 집계 ...")
    data = collect()
    dates = [d for d, _ in data]
    cons = np.array([p["consistency_ratio"] for _, p in data])
    mean, std = cons.mean(), cons.std()

    labels = [formality.FORMAL, formality.INFORMAL_POLITE, formality.BANMAL, formality.NEUTRAL]
    colors = {"격식존댓말": "#1f4e79", "비격식존댓말": "#5aa0d6",
              "반말": "#e07b54", "중립": "#bdbdbd"}
    # 파일별 비율(%) 구성
    comp = {lab: [] for lab in labels}
    for _, p in data:
        total = sum(p["counts"].values())
        for lab in labels:
            comp[lab].append(p["counts"][lab] / total * 100 if total else 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    # (좌) consistency_ratio 분포
    ax1.hist(cons, bins=np.arange(62, 82, 2), color="#5aa0d6", edgecolor="white")
    ax1.axvline(mean, color="crimson", ls="--", lw=1.5)
    ax1.text(mean + 0.3, ax1.get_ylim()[1] * 0.88,
             f"평균 {mean:.1f}%\n±{std:.1f}", color="crimson", fontsize=9)
    ax1.set_title("consistency_ratio 분포 (15강의)")
    ax1.set_xlabel("consistency_ratio (%)")
    ax1.set_ylabel("강의 수")
    ax1.grid(alpha=0.3, axis="y")

    # (우) 강의별 말투 구성 누적막대
    x = np.arange(len(dates))
    bottom = np.zeros(len(dates))
    for lab in labels:
        vals = np.array(comp[lab])
        ax2.bar(x, vals, bottom=bottom, label=lab, color=colors[lab], width=0.8)
        bottom += vals
    ax2.set_title("강의별 말투 구성 (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(dates, rotation=90, fontsize=7)
    ax2.set_ylabel("문장 비율 (%)")
    ax2.set_ylim(0, 100)
    ax2.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))

    fig.suptitle("항목 3 언어 일관성 — EDA  (2026-06-15, 이수민)", fontsize=13, y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")

    viol = [p["violation_count"] for _, p in data]
    print(f"저장: {os.path.normpath(OUTPUT_PNG)}")
    print(f"consistency_ratio: 평균 {mean:.1f}  범위 {cons.min():.1f}~{cons.max():.1f}  std {std:.1f}")
    print(f"violation_count: 평균 {np.mean(viol):.0f}  범위 {min(viol)}~{max(viol)}")


if __name__ == "__main__":
    main()
