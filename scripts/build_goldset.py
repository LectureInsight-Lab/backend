"""scripts/build_goldset.py — EVAL_MODEL(Gemini 2.5 Pro)로 항목 16·17 1차 골든셋 재라벨.

골든셋 라벨러 = EVAL_MODEL(settings.eval_model = models/gemini-2.5-pro).
출력: data/gold/item{16,17}_{MM-DD}.json  (항목 2 × 일자 5 = 10개, per-date)
부산물:
  - 기존 Claude 라벨은 data/gold/_claude_ref/ 로 백업(IAA 비교용)
  - Claude vs Gemini 일치도(IAA) 콘솔 출력

실행:
  cd backend
  GEMINI_API_KEY=... python scripts/build_goldset.py            # 16·17 전체
  GEMINI_API_KEY=... python scripts/build_goldset.py 16 02-09   # 일부만

전제: pip install google-genai, kiwipiepy (이 환경엔 미설치 → 키 있는 환경에서 실행).
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from google import genai
from google.genai import types

from app.core.config import settings
from app.preprocessing import sentencizer
from app.preprocessing.preprocessor import parse

EVAL_MODEL = settings.eval_model                      # models/gemini-2.5-pro
_API_KEY = os.environ.get("GEMINI_API_KEY") or settings.api_key
CLIENT = genai.Client(api_key=_API_KEY)

STT_GLOB = "../data/강의 스크립트/*.txt"
GOLD_DIR = "data/gold"
CLAUDE_REF = os.path.join(GOLD_DIR, "_claude_ref")
DATES = ["02-09", "02-10", "02-11", "02-12", "02-13"]
BATCH = 30                                             # 한 호출당 문장 수 (작을수록 진행 피드백 잦음)

# ── 후보 net (Claude 라벨 때와 동일해야 idx 매칭/IAA 성립) ──────────────
NET = {
    16: re.compile(
        r"되셨|됐(어|죠|나|어요)|되시(나요|죠|어요)|아시겠|알겠(어|죠|나|지)|맞(죠|나요|으시|지요|어)|"
        r"괜찮(으세요|으시|나요|죠)|보이(시죠|시나요|세요|죠|나요)|따라\s*오(시|셨)|질문\s*(있|없)|"
        r"이해\s*(가|되|하|갔|간|돼)\S{0,3}(나요|죠|어요|시|셨|가요|가죠|갔죠|간다|돼요|되나)|"
        r"헷갈리(시|나)|모르(시|겠어|겠죠)|아셨|을까요|ㄹ까요|는지\s*(아|모|보|알)|여기까지|이거\s*됐|됐냐|"
        r"알아\s*(들|보)\S*(나요|죠|어요)"
    ),
    17: re.compile(
        r"(해|써|짜|풀어|작성해|입력해|구현해|만들어|실행해|돌려|테스트해|넣어|쳐|확인해|로드해|체크해|"
        r"생각해|추출해|사용해|인서트해)\s*보|직접\s|혼자\s|스스로|한번\s*(해|써|짜|풀|만들|작성|구현|보|돌려)|"
        r"따라\s*(해|치)|각자\s"
    ),
}

# ── 라벨링 지침 (우리가 데이터로 확정한 규칙) ──────────────────────────
RUBRIC = {
    16: (
        "당신은 한국어 강의 STT 문장에서 '이해 확인 질문'(항목16)을 판별하는 평가자입니다.\n"
        "정의: 강사가 학생에게 이해/완료 여부를 묻는 발화 = CHECK(true).\n"
        "규칙:\n"
        "- true: 시-높임 확인(되셨어요/되시나요/하셨어요/오셨어요/보이시죠/괜찮으세요/이해 가셨어요),"
        " 의문형 확인(됐을까요/됐습니까/됐냐/알겠죠/맞죠/이해 가죠/이해되셨어요), '여기까지 되셨어요/오셨어요'\n"
        "- false: 단순 상태서술 '됐어요/왔어요'(높임·의문 없이 코드/결과가 됐다는 서술),"
        " 1인칭 '모르겠어'(강사 본인), 설명체, 절차 지시 명령형(확인하세요/오세요)\n"
        '각 문장의 idx에 대해 true/false. JSON {"labels": {"<idx>": true, ...}} 형식으로만 응답.'
    ),
    17: (
        "당신은 한국어 강의 STT 문장에서 '참여 유도'(항목17)를 strong/weak/no 3단계로 판별하는 평가자입니다.\n"
        "STT라 어미가 잘릴 수 있다('해보세'→'해보세요', '써보세'→'써보세요'). 잘린 명령형도 명령형으로 본다.\n"
        "판정은 '서법(누가 행위자인가)'으로 한다. 핵심: 직접 명령만 strong, 전언/서술/시범은 절대 strong 아님.\n"
        "- 'strong' = 학생을 향한 **직접 명령**(2인칭 명령형): ~해보세요/풀어봐요/넣어봐/써봐/해봐/~어라/바꿔주세요"
        " (STT로 잘린 '해보세·써보세' 포함). '지금 학생이 직접 하라'.\n"
        "- 'weak' = **청유형 과제 제시**: ~해보자/풀어보자/구현해보자/출력해보자 ('우리 ~하자'로 학생에게 과제를 던짐)."
        " 단 강사 본인 시범 청유는 제외.\n"
        "- 'no' = 그 외 전부:\n"
        "   · 강사 본인 시범: 제가/내가 ~할게요/해볼게요/해보도록 하겠습니다/보여드릴게요/연습\n"
        "   · 전언·인용(남에게 시키라고 *전달*, 직접 명령 아님): ~라고 해요/~(으)래요/~보시래요/~시라고 (예: '테스트해보시라고 해요')\n"
        "   · 서술·진행(명령이 아니라 동작 *묘사*, 연결어미): ~해 보고/~넣어보고/~하는 거야/~보는 거야/~해보면 (예: '넣어보는 거야', '입력해 보고')\n"
        "   · 단순 관찰 지시(그냥 '보세요'=화면 보기), 기술용어 오탐(직접 호출), 무관\n"
        '각 문장의 idx에 strong/weak/no. JSON {"labels": {"<idx>": "strong", ...}} 형식으로만 응답.'
    ),
}
LABEL_KEY = {16: "is_check", 17: "tier"}


def _sentences(date: str):
    f = next(x for x in sorted(glob.glob(STT_GLOB)) if date in x)
    return os.path.basename(f), sentencizer.build_sentences(parse(open(f, encoding="utf-8").read()), use_morph=True)


def _label_batch(item: int, batch: list[tuple[int, str]]) -> dict:
    numbered = "\n".join(f"{i}\t{t}" for i, t in batch)
    prompt = RUBRIC[item] + "\n\n문장 목록 (idx<TAB>text):\n" + numbered
    resp = CLIENT.models.generate_content(
        model=EVAL_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json"),
    )
    return json.loads(resp.text).get("labels", {})


def _normalize(item: int, raw):
    if item == 16:
        return bool(raw) if not isinstance(raw, str) else raw.strip().lower() in ("true", "y", "yes", "check")
    v = str(raw).strip().lower()
    return v if v in ("strong", "weak", "no") else "no"


class _Heartbeat:
    """블로킹 API 호출 동안 경과 초를 같은 줄에 실시간 표시 (진행 중임을 보여줌)."""

    def __init__(self, label: str):
        self.label = label
        self._stop = threading.Event()
        self._t: threading.Thread | None = None

    def __enter__(self):
        def run():
            start = time.time()
            while not self._stop.wait(1.0):
                print(f"\r      {self.label} … {time.time() - start:4.0f}s 경과", end="", flush=True)

        self._t = threading.Thread(target=run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._t:
            self._t.join(timeout=0.2)
        print("\r" + " " * 72 + "\r", end="", flush=True)   # 줄 지우기


def build(item: int, date: str):
    print(f"  [item{item} {date}] 문장 분리 중 (Kiwi 로딩+분리, 수 초)...", flush=True)
    fname, sents = _sentences(date)
    cand = [(i, s.text) for i, s in enumerate(sents) if NET[item].search(s.text)]
    n_batch = (len(cand) + BATCH - 1) // BATCH
    print(f"  [item{item} {date}] 후보 {len(cand)}개 → {EVAL_MODEL} 라벨링 (배치 {n_batch}개)", flush=True)
    labels: dict[int, object] = {}
    for bi, k in enumerate(range(0, len(cand), BATCH), 1):
        chunk = cand[k:k + BATCH]
        t0 = time.time()
        with _Heartbeat(f"배치 {bi}/{n_batch} ({len(chunk)}문장) Pro 응답 대기"):
            got = _label_batch(item, chunk)
        for i, _ in chunk:
            labels[i] = _normalize(item, got.get(str(i), False if item == 16 else "no"))
        done = [labels[i] for i, _ in chunk]
        if item == 16:
            tally = f"check {sum(1 for v in done if v)}/{len(done)}"
        else:
            tally = " / ".join(f"{t} {sum(1 for v in done if v == t)}" for t in ("strong", "weak", "no"))
        print(f"    ✓ 배치 {bi}/{n_batch} 완료 ({time.time() - t0:4.0f}s) — {tally}", flush=True)

    key = LABEL_KEY[item]
    if item == 16:
        n_pos = sum(1 for v in labels.values() if v)
        meta_counts = {"n_gold": n_pos}
    else:
        meta_counts = {"n_strong": sum(1 for v in labels.values() if v == "strong"),
                       "n_weak": sum(1 for v in labels.values() if v == "weak")}
    doc = {
        "_meta": {
            "task": f"항목{item} 1차 후보군 gold",
            "file": fname,
            "rule": RUBRIC[item].splitlines()[1],
            "annotator": f"EVAL_MODEL={EVAL_MODEL}",
            **meta_counts,
        },
        "candidates": [{"idx": i, "text": t, key: labels[i]} for i, t in cand],
    }
    out = os.path.join(GOLD_DIR, f"item{item}_{date}.json")
    json.dump(doc, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    iaa = _compare_claude(item, date, labels, key)
    print(f"  item{item} {date}: 후보 {len(cand)} → {meta_counts}  {iaa}")


def _compare_claude(item: int, date: str, gemini: dict, key: str) -> str:
    """기존 Claude 라벨(백업본)과 일치도(IAA) 계산."""
    ref = os.path.join(CLAUDE_REF, f"item{item}_{date}.json")
    if not os.path.exists(ref):
        return "(claude ref 없음)"
    cl = {c["idx"]: c.get(key) for c in json.load(open(ref, encoding="utf-8"))["candidates"]}
    common = [i for i in gemini if i in cl]
    if not common:
        return "(idx 불일치)"
    agree = sum(1 for i in common if gemini[i] == cl[i])
    return f"IAA(Claude vs Gemini) {agree}/{len(common)} = {agree/len(common)*100:.0f}%"


def main():
    os.makedirs(CLAUDE_REF, exist_ok=True)
    # 기존 Claude 라벨 백업 (최초 1회만)
    for p in glob.glob(os.path.join(GOLD_DIR, "item1*_*.json")):
        dst = os.path.join(CLAUDE_REF, os.path.basename(p))
        if not os.path.exists(dst):
            shutil.copy(p, dst)

    args = sys.argv[1:]
    items = [int(args[0])] if args else [16, 17]
    dates = [args[1]] if len(args) > 1 else DATES
    print(f"EVAL_MODEL = {EVAL_MODEL}")
    for item in items:
        print(f"[항목 {item}]")
        for d in dates:
            build(item, d)


if __name__ == "__main__":
    main()
