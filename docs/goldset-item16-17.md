# 골든셋 구축 — 항목 16·17 (이해 확인 질문 / 참여 유도)

> 담당: 이수민 · 최종 갱신 2026-06-28
> 대상: 항목 16(이해 확인 질문) · 17(참여 유도)의 **1차 골든셋**(후보군 추출 / recall 측정용)
> 관련: [scoring-bands.md](scoring-bands.md) · [items_2_3_12_16_17.md](items_2_3_12_16_17.md)

---

## 0. 왜 골든셋인가
채점 로직(regex·임계값·밴드)이 전부 손으로 만든 것이라, "잘 잡겠지"를 **수치로 증명**해야 한다.
- 정확도(recall/precision) 측정 → 신뢰 근거 / 임계값·패턴 튜닝의 객관적 기준
- 회귀 방지, 오류 진단, 방법 비교
> 항목 2·3은 검증 경로 존재(3 = honorific gold, 2 = Kiwi 태깅 감사). **미구축 = 12·13·16·17**. 본 문서는 16·17.

---

## 1. 방법론
- **2단계**: ① 1차 후보군 추출(recall) → ② 2차 채점(scoring, 미착수)
- **baseline(비교군) 대비 향상 측정**: baseline = 루브릭 spec 예시 regex / 비교 대상 = 우리 데이터튜닝 regex(production)
- **생성 파이프라인**: STT → Kiwi 문장 인덱싱 → 넓은 net으로 후보 surfacing → **EVAL_MODEL이 라벨** → per-date JSON
- **단위**: Kiwi 문장 (gold·detector 동일 단위 → 공정 비교)
- **대상**: 5개 파일 (2026-02-09 ~ 02-13)

### gold 라벨러 = EVAL_MODEL
- **`EVAL_MODEL = models/gemini-2.5-pro`** (config.py / .env). production 채점용 `LLM_MODEL=gemini-2.5-flash`와 **분리**:
  강한 모델로 정답을 만들고 cheaper flash production을 채점 → 서로 달라 순환 없음.
- **2-annotator 검증**: 초기엔 Claude가 1차 라벨 → 이후 Gemini Pro로 재라벨하고 **둘의 일치도(IAA)** 로 신뢰도 확인.
  (`data/gold/_claude_ref/`에 Claude 라벨 보존)

---

## 2. 핵심 원칙
> **모호한 범주에 binary를 강제하지 말고, 등급 라벨로 담는다.**
> **1차(텍스트 = 의도) ≠ 2차(갭 = 행동)를 분리한다.**
- 16의 "됐어요", 17의 "해보자"가 둘 다 회색지대 → 이 원칙으로 수렴
- 정보 손실 방지 + 임계 결정을 나중으로 미뤄도 재라벨 불필요

---

## 3. 라벨링 규칙 (EVAL_MODEL 프롬프트 = `scripts/build_goldset.py`)

### 항목 16 — `is_check ∈ {true, false}`
- **true(CHECK)**: 시-높임 확인(되셨어요/되시나요/하셨어요/오셨어요/보이시죠/괜찮으세요/이해 가셨어요),
  의문형 확인(됐을까요/됐습니까/됐냐/알겠죠/맞죠/이해 가죠), '여기까지 되셨어요/오셨어요'
- **false**: bare 상태서술 '됐어요/왔어요'(높임·의문 없음), 1인칭 '모르겠어', 설명체, 절차 지시(확인하세요)

### 항목 17 — `tier ∈ {strong, weak, no}` (서법 기준)
- **strong** = 학생을 향한 **직접 명령**(2인칭 명령형): 해보세요/풀어봐요/넣어봐/써봐/~어라/바꿔주세요.
  STT로 어미 잘린 명령형(해보세/써보세 = 해보세요/써보세요) 포함
- **weak** = **청유형 과제 제시**: 해보자/풀어보자/구현해보자('우리 ~하자'로 과제를 던짐) − 강사 본인 시범 제외
- **no** = 강사 본인 시범(제가/내가 ~할게요/하겠습니다/보여드릴게요/연습), 전언("~라고 해요/~래요/~시라고"),
  서술·진행("~하는 거야/~보는 거야/~해 보고"), 단순 관찰("그냥 보세요"), 기술용어 오탐(직접 호출)
- 핵심: **직접 명령만 strong. 전언·서술·시범은 절대 strong 아님.**

---

## 4. 핵심 발견 · 결정 (데이터로)

| # | 발견 | 결정 |
|---|---|---|
| 1 | 루브릭 spec 예시가 실강의에 **0건** (이해하셨나요 등) | baseline recall 0% — 우리 가설 입증 |
| 2 | 16 "됐어요" 이중성: 시-높임 222 vs bare 212(41%가 상태서술) | **시-높임/의문형만 CHECK** |
| 3 | 16 production: recall 갭(이해 가죠/됐습니까) + precision 누수(bare 됐어요) | detector 개선 리스트 |
| 4 | 17 production recall 심각(@strong 28%) — _ACTION 누락·요 강제·STT 잘림 | detector 개선 |
| 5 | 17 청유형 "해보자"가 명령형의 **2.9배** (강사 주력), production은 제외 | **3-tier(strong/weak/no)** |
| 6 | 청유 학생-vs-강사 **자동 분리 불가**(75% 모호), 갭 신호도 약함 | 음성 필터(강사 시범 제외) + 등급 라벨 |
| 7 | gold 라벨러로 **Gemini 2.5 Pro(EVAL_MODEL)** 결정 | Claude 1차 → Pro 재라벨 + IAA |
| 8 | 17 초기 IAA 69%, 불일치가 체계적(전언·서술→strong 오탐) | **규칙을 서법 기준으로 조임** → IAA 84% |
| 9 | 잔여 불일치(02-09 기준 ~19): 2/3는 Gemini가 더 정확(Claude regex 한계), ~6만 진짜 애매 | 사람 판정은 그 소수만 |

---

## 5. 결과

### 5-1. IAA (Claude vs Gemini Pro)
| 파일 | 16 | 17 |
|---|---|---|
| 02-09 | 97% | 84% |
| 02-10 | 95% | 84% |
| 02-11 | 99% | 85% |
| 02-12 | 95% | 88% |
| 02-13 | 94% | 83% |
| **범위** | **94~99% (거의 완벽)** | **83~88% (좋음·일관)** |

→ 16은 규칙이 깔끔해 거의 완전 일치, 17은 규칙 조임 후 5파일 전반 수렴. **신뢰 가능한 골든셋**.

### 5-2. gold 규모 (EVAL_MODEL 라벨)
- **16**: is_check=true 합 **227** (57/50/47/28/45)
- **17**: strong 합 **89** / weak 합 **139** (strong+weak 228)

### 5-3. detector recall (공식 Gemini gold 기준)
| 항목 | baseline(spec) | production |
|---|---|---|
| **16 이해확인** | 0% | **66%** (150/227) |
| **17 참여유도** | — | **@strong 28%** (25/89) · **@strong+weak 11%** (25/228) |

- 16: 양호·안정(파일별 61~72%). 놓침은 동일 유형(이해 가죠/됐습니까/됐냐) → 패턴 보강 여지.
- 17: **심각한 recall 갭**. gold가 정확해질수록(Gemini가 명령형/STT-cut 포함) production 한계가 더 또렷 → "정의가 현실과 어긋난" 문제.

---

## 6. 산출물 · 파일 구조
```
backend/data/gold/
  item16_02-09.json … item16_02-13.json     # 항목16 × 5일
  item17_02-09.json … item17_02-13.json     # 항목17 × 5일  (총 10개, per-date)
  _claude_ref/                               # Claude 1차 라벨 백업(IAA용)
  .gitkeep
```
JSON 포맷:
```json
{ "_meta": { "task", "file", "rule", "annotator": "EVAL_MODEL=models/gemini-2.5-pro", "n_gold"/"n_strong"/"n_weak" },
  "candidates": [ { "idx", "text", "is_check"(16) | "tier"(17) }, ... ] }
```

### 보안: gold 파일은 git 추적 제외
보안서약상 실데이터 커밋 금지 → `.gitignore`에 추가(경로는 `.gitkeep`로 유지):
```
data/gold/*
data/gold/_claude_ref/*
!data/gold/.gitkeep
```

---

## 7. 재현 / 실행
```bash
cd backend
pip install google-genai kiwipiepy            # 최초 1회
export GEMINI_API_KEY=...                       # 또는 .env 의 API_KEY
python scripts/build_goldset.py                 # 16·17 전체(10개) 재라벨 + IAA
python scripts/build_goldset.py 17 02-09        # 일부만 (항목 17, 02-09)
```
- 라벨러 모델: `settings.eval_model`(.env `EVAL_MODEL`) = `models/gemini-2.5-pro`
- 진행 로그(heartbeat): Pro 응답 대기 경과초·배치별 집계 실시간 표시
- 기존 Claude 라벨은 자동 `_claude_ref/` 백업 후 IAA 비교

---

## 8. 다음 단계 (TODO)
- [x] 16·17 1차 골든셋 5파일 × 2 = 10개 (EVAL_MODEL 라벨)
- [x] EVAL_MODEL(gemini-2.5-pro) config/.env 명시 + 재라벨 스크립트 + 진행 로그
- [x] IAA 검증(16: 94~99% / 17: 83~88%) · detector recall(16: 66% / 17: 28%·11%)
- [x] gold 파일 git 추적 제외(.gitignore) + 경로 유지(.gitkeep)
- [ ] 잔여 불일치 사람 판정 (특히 17의 진짜 애매 ~6/파일) → gold 최종 확정
- [ ] detector 패턴 개선(4절 리스트) 적용 후 recall 재측정 → 향상폭 확정
- [ ] 2차(채점) gold 설계
- [ ] 항목 12·13 골든셋
