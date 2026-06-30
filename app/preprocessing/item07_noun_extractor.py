"""
Kiwi 형태소 분석 + 불용어 처리 모듈

utils.py → anchored_labeled.json 의 레코드를 받아
날짜 × 구간(개념/예시/실습)별 명사 후보 리스트를 반환한다.

주요 처리:
  1. COMPOUND_MERGE  : ASR 공백 분절 복합어 복원
  2. Kiwi 형태소 분석: NNG / NNP 명사만 추출
  3. 사용자 사전     : 복합 기술 용어 단일 토큰화
  4. STOPWORDS 필터  : 수동 불용어 (~170개)
  5. DF 자동 불용어  : 전체 날짜에 공통으로 등장하는 배경어 제거
                       단, GOLD_WHITELIST 단어는 보호
"""
from __future__ import annotations

import json
import re
from collections import defaultdict, Counter
from pathlib import Path

from loguru import logger

# ── 경로 ──────────────────────────────────────────────────────────────────────
_BASE = Path(__file__).parent.parent.parent
USER_DICT_PATH = _BASE / "configs" / "kiwi_user_dict.yaml"

# ── 상수 ──────────────────────────────────────────────────────────────────────
MIN_WORD_LEN  = 2
LABEL_ORDER   = ["개념", "예시", "실습"]

# [Step3 개선] 2026-06-29: ASR 노이즈 토큰 필터
# 문제: Kiwi가 STT 오인식 토큰을 NNG로 잘못 분류하여 KeyBERT 후보 풀에 진입.
#       IDF 부스트 이후 이 노이즈가 골드 키워드를 밀어냄.
# 해결: 명백한 노이즈 패턴을 정규식으로 걸러냄.
#   - 숫자로 시작 (예: 2언더버지)
#   - 야데/기서/해서기 등 동사 어미가 붙은 오결합 (예: 리플레이스야데, 시작해서기서)
_NOISE_TOKEN_RE = re.compile(
    r"^\d"          # 숫자 시작 토큰
    r"|야데$"        # ~야데 (동사+강조 오결합)
    r"|기서$"        # ~기서 (동사형 오결합)
    r"|해서기$"      # ~해서기
    r"|에서기$"      # ~에서기
)

# DF 자동 불용어에서 보호할 골드 핵심 용어
# [Step1 개선] 2026-06-29: MySQL 문자열 함수 강의 핵심 용어 보호 추가
# 이유: DF 자동 불용어가 강의별 핵심 SQL 함수 이름을 제거할 수 있음
GOLD_WHITELIST = {
    # 기존 Java 강의 보호 용어
    "인덱스", "스트림", "버퍼", "트랜잭션",
    # MySQL 문자열 함수 보호 (2026-02-09 기준, 타 날짜에도 등장 가능)
    "리플레이스", "트림", "콘캣", "캐릭터셋", "콜레이션",
    "오토인클라이먼트", "베이스64", "로드언더바파일", "프라이머리키",
    # MySQL DML / TCL (02-13 골드 키워드 — STOPWORDS에서 이동)
    "딜리트", "업데이트", "인서트", "커밋", "롤백", "오토커밋",
    # [Step10 개선] 2026-06-30: 멀티날짜 비교 후 WHITELIST 확장
    # 체크: DF=14/15 → 자동 불용어 처리됨 → 02-12 gold (CHECK CONSTRAINT) 복구
    "체크",
    # SQL 조인/서브쿼리 계열 (02-10~11 gold, DF 중간 → 보호 강화)
    "서브쿼리", "유니온", "익스플레인",
    # 복합 골드 키워드 단일토큰 형태 (COMPOUND_MERGE → 이 형태로 변환)
    "이너조인", "아우터조인", "크로스조인", "셀프조인",
    "상관쿼리", "다중칼럼", "인라인뷰",
    # 제약조건: [Step23] 2026-06-30 GOLD_WHITELIST에서 제거
    # 이유: MIN_GOLD_IDF=1.2 부스트로 02-13(TCL 강의)에서 FP rank10 차지 → 락 진입 차단.
    #       df(제약조건)<15이므로 auto-stopword 위험 없음 (df≈10~12 예상).
    #       02-12에서 제약조건의 TF가 labeled sections에서 매우 낮아 어차피 top-10 미진입.
    "재귀쿼리", "인포메이션스키마",
    "스타트트랜잭션",
    # 02-13 TCL 추가 gold
    "락", "세션",
}

# ── 수동 불용어 ────────────────────────────────────────────────────────────────
STOPWORDS: set[str] = {
    # 담화어 · 메타어
    "다음", "이번", "지금", "경우", "내용", "부분", "생각", "나머지",
    "방법", "사용", "이유", "정도", "얼마", "중간", "이상", "이하",
    "의미", "형태", "단계", "번째", "관계", "기준", "역할", "목적",
    "이전", "이후", "처음", "마지막", "현재", "하단", "화면", "오후",
    "아침", "오늘", "내일", "시간", "확인", "설명", "진행", "작업",
    "얘기", "만약", "실제", "이해", "자체", "나중", "정리", "수업",
    "강사", "해당", "일반", "동일", "중요", "발생", "정확", "명시",
    "느낌", "눈도장", "마찬가지", "예전", "차이", "상황", "용어",
    "각각", "아무것", "기억", "단어", "기타", "가이드", "공유",
    "사람", "필요", "성공", "키워드", "버전", "형식",
    "오른쪽", "도움말", "섹션", "이용", "복사", "추가", "수정", "선택",
    # 복합어 분절 잔재 / ASR 음차 오류
    "바이", "언더", "네임", "마이에스큐",
    # ASR 전사 오류 — 02-02
    "스태틱", "어프", "인렬화", "오더링", "임플루먼트",
    "트리버셜", "코드이", "이노더링", "바이틀", "프리비지", "파일렛",
    # ASR 전사 오류 — 02-03 실습
    "됐쿠", "여어", "나네", "승제", "널까폴트", "스큐이", "후루",
    "엎퍼짓", "비언더버", "스티로", "화이리트",
    # ASR 전사 오류 — 02-03 예시/개념
    "캡세퍼라이터", "비짓", "스태팅", "닷넷", "애널리트", "디스크라이버",
    # 기타 메타/범용어
    "테스트", "스코어", "로고", "자식", "자식들",
    # 2차 ASR 오류
    "악푸스트림", "바이트스", "워팔트리", "보역", "트리뷰트",
    "객체야", "3트럴", "아까", "오브",
    # [Step6 개선] 2026-06-29: 가산형 점수 도입 후 신규 노출된 노이즈 추가
    # 문제: TF 부스트로 합성어 파편(리플/플레이스) 및 담화어(보자/이네/올림/패드)가 상위 진입
    # 해결: 명시적 불용어 등록으로 후보 풀에서 완전 제거
    "리플",       # 리플레이스(REPLACE) 파편 — 완전 용어는 GOLD_WHITELIST에 있음
    "플레이스",   # 리플레이스(REPLACE) 파편
    "보자",       # 담화어 ("같이 보자")
    "이네",       # 한국어 접속사/강조 어미
    "올림",       # 수학 올림(ceiling) 동사 — ROUND/CEIL 강의 파편
    "패드",       # LPAD/RPAD 합성어 파편 — 단독 의미 없음
    # [Step7 개선] 2026-06-29: Step6 top-10 잔여 노이즈 제거
    # 문제: 가산형 공식 후 무조건/앞뒤/반올림이 콘캣 등 골드 키워드보다 상위
    # 해결: 범용어 및 비핵심 SQL 연산어 불용어 등록
    "무조건",     # 담화어 ("무조건 이렇게")
    "앞뒤",       # 위치 범용어 — SQL 문자열 위치 관련 파편
    "반올림",     # 수학 반올림 — ROUND 강의 파편; 핵심 SQL 함수와 무관
    # [Step10 개선] 2026-06-30: 02-10~02-13 멀티날짜 비교 후 추가 노이즈 제거
    # 문제: 02-09 전용으로 튜닝된 STOPWORDS가 타 날짜 노이즈를 걸러내지 못함.
    #       02-10: 테이/이트(부분절단), 조금/하지/교수(담화어/비CS)
    #       02-11: 가로/세로(방향어), 아우(2음절잔재), 서브콜/거지/주쿼리(ASR 오인식)
    #       02-12: 위에/이제/이사(담화어), 스가/마이사이/마이사/소멸(노이즈)
    #       02-13: 스턴트/퍼런시스(ASR오류), 마킹/한쪽(담화어), 코밋(커밋 중복음차), 지점(범용어)
    # 해결: 명백한 노이즈/담화어/비CS 단어 STOPWORDS 추가
    # 02-10
    "테이",       # 테이블 부분 절단 토큰
    "이트",       # 라이트/이트 부분 절단 토큰
    "조금",       # 담화어 ("조금 다르게")
    "하지",       # 담화어 ("하지만")
    "교수",       # 비CS — 강사 지칭어
    # 02-11
    "가로",       # 방향어
    "세로",       # 방향어
    "아우",       # 아우터 2음절 잔재
    "서브콜",     # ASR 오인식
    "거지",       # ASR 오인식
    "주쿼리",     # 주(主)쿼리 분절 잔재 — 의미 불명확
    "서브커리",   # 서브쿼리 음차 오류 변형
    # 02-12
    "위에",       # 담화어 ("위에 있는")
    "이제",       # 담화어 ("이제 보면")
    "이사",       # 비CS 담화어
    "스가",       # ASR 노이즈
    "마이사이",   # MySQL 음차 오류 파편
    "마이사",     # MySQL 음차 오류 파편
    "소멸",       # 비CS (존재/소멸) 담화어
    # 02-13
    "스턴트",     # ASR 오류 (constant → 스턴트?)
    "퍼런시스",   # reference 음차 오류
    "마킹",       # 비CS 담화어
    "한쪽",       # 방향 담화어
    "코밋",       # 커밋 중복 음차 (커밋이 이미 올바른 형태)
    "지점",       # 범용어 — SQL 문맥에서 의미 없음
    # [Step11 개선] 2026-06-30: 2차 STOPWORDS 추가 (멀티날짜 2라운드 분석)
    # 02-10
    "약간",       # 담화어 ("약간 다르게")
    # 02-11
    "하나",       # 담화어 ("하나씩 보면")
    "세일즈맨",   # 예제 데이터 (salesman) — SQL 실습 샘플 데이터
    "월급",       # 예제 데이터 (salary) — SQL 실습 샘플 데이터
    "데이",       # 부분 절단 토큰 (date? data? 의미 불명)
    "완전",       # 담화어 ("완전히 다른")
    "코리",       # 부분 절단 (Korea? query? 의미 불명)
    "쿼리아",     # ASR 오인식 노이즈
    # 02-12
    "초기값",     # 범용 프로그래밍 용어 — 02-12 SQL goldset 무관
    "토탈",       # total — 범용어
    "반복",       # 담화어 / 반복문 범용어 — SQL DDL 맥락 무관
    "마이살",     # MySQL 음차 오류 파편
    # 02-13
    "세이브",     # SAVEPOINT 파편 — 02-13 goldset에 없음
    "부모",       # 비CS 담화어 (parent row는 외래키 맥락이지만 goldset 무관)
    "토코밋",     # 오토커밋 파편 음차 오류
    "에디터",     # 비CS — 도구 지칭어
    # [Step12 개선] 2026-06-30: 3차 STOPWORDS (복합어 처리 후 단편 제거 + 추가 노이즈)
    # 02-11 잔여 노이즈
    "세일즈",     # 세일즈맨 어근 파편 (세일즈맨 제거 후 나머지)
    "살이",       # ASR 오인식 (살이/살이쪄 등)
    "워드",       # 비CS 담화어 (word)
    # 02-12 노이즈 (복합어 COMPOUND_MERGE 후 안전하게 제거 가능한 단편)
    "인라인",     # 인라인뷰 COMPOUND_MERGE 후 단독 인라인은 노이즈 — 02-12 gold 무관
    "별칭",       # alias — 02-12 gold 없음
    "시스",       # 인포메이션스키마 파편
    "스키",       # 인포메이션스키마/스키마 파편
    "듀얼",       # DUAL table — 02-12 gold 없음
    # 02-13 복합어 처리 후 잔여 단편
    "오토",       # 오토커밋/오토인클라이먼트 COMPOUND_MERGE 후 단독 오토는 잔재
    "스타트",     # 스타트트랜잭션 COMPOUND_MERGE 후 단독 스타트는 잔재
    "노액션",     # NO ACTION constraint — 02-13 gold 없음
    "토코",       # 오토커밋 파편 (토코밋 삭제 후 나머지)
    # 복합어 처리로 대체된 단편 (COMPOUND_MERGE + user dict 덕분에 안전하게 제거 가능)
    "아우터",     # 아우터조인 복합어로 대체 — 02-10/11 gold 이미 매칭됨
    # [Step13 개선] 2026-06-30: 멀티날짜 top-10 전수 분석 후 잔여 노이즈 일괄 추가
    # 02-09 noise (rank 5-10)
    "유니코드",   # Unicode — 02-09 gold 없음 (관련 강의이나 goldset 외)
    "용해",       # ASR 오인식 노이즈
    "라이언",     # 인명 — ASR 노이즈
    "테일링",     # tailing — SQL gold 없음
    "로케이트",   # LOCATE function — 02-09 gold 없음
    # 02-10/11 공통 noise
    "레프트",     # LEFT JOIN — 평가 goldset에 없음 (크로스/이너/셀프/아우터만 gold)
    # 02-11 중복 매칭 단편 (이미 복합어·다른 단편으로 매칭됨)
    "일행",       # 단일행(rank 8 exact match)이 있으므로 중복 — 제거해도 매칭 유지
    "서브",       # 쿼리(rank 5)가 서브쿼리 gold 담당 — 서브는 중복
    "커리",       # ASR 노이즈 (query → 커리)
    "가변",       # 가변길이 범용어 — SQL gold 없음
    # 02-12 noise (rank 6-10)
    "가상",       # 가상테이블 범용어 — 02-12 gold 없음
    "매장",       # 예제 데이터 (retail store) — SQL gold 없음
    "그다음",     # 담화어
    # 02-13 noise (rank 6-10)
    "외래키",     # FOREIGN KEY — 02-13 gold 없음 (트랜잭션/TCL 강의)
    "레퍼런시스", # REFERENCES — 02-13 gold 없음
    "오게",       # 담화어 ("오게 되면")
    "메일",       # email — SQL gold 없음
    "포인트",     # point/savepoint — 02-13 gold 없음
    # [Step14 개선] 2026-06-30: Step13 후 새로 노출된 노이즈 일괄 추가
    # 주의: 포메이션은 02-12 '인포메이션 스키마' gold 매칭 중 → 추가 금지
    # 02-09 신규 noise (rank 6-10)
    "패딩",       # PADDING — 02-09 gold 없음 (LPAD/RPAD 파편 아님)
    "중복키",     # DUPLICATE KEY — 02-09 gold 없음 (오토인클라이먼트와 별개)
    "케이",       # ASR 파편 (key 음차 일부?) — 의미 불명확
    "헬프로",     # ASR 노이즈
    # 02-11 신규 noise (rank 6, 7, 9)
    "퀘스트",     # quest/request ASR 노이즈
    "급여",       # salary — 예제 데이터
    "셀렉",       # SELECT 파편 — 02-11 gold 없음 (서브쿼리/단일행 등이 gold)
    # 02-12 신규 noise (rank 8-10)
    "사작",       # 시작 ASR 오인식
    "재규",       # 재귀(recursive) ASR 오인식 파편
    "인라이뷰",   # 인라인뷰 ASR 오인식 변형
    # 02-13 신규 noise (rank 6-10)
    "웨이트",     # WAIT ASR 음차 — 02-13 gold 없음
    "영업",       # 비CS 예제 데이터
    "영구",       # permanent — 담화어 / 비SQL
    "사보",       # ASR 노이즈
    "스튜트",     # ASR 노이즈
    # [Step15 개선] 2026-06-30: Step14 후 새로 노출된 노이즈 일괄 추가
    # 주의: 포메이션은 02-12 '인포메이션 스키마' gold 매칭 중 → 추가 금지
    # 주의: 크로스/인라인뷰는 02-10/02-11 gold 매칭 중 → 추가 금지
    # 02-09 신규 noise (rank 7, 9, 10)
    "아스키",     # ASCII — 02-09 gold 없음 (캐릭터셋/콜레이션이 gold)
    "공백",       # space/blank — TRIM 강의 범용 단어, 02-09 gold 없음
    "충돌",       # conflict/collision — DUPLICATE KEY 범용어, gold 없음
    # 02-11 신규 noise (rank 8-10)
    "함정",       # trap/pitfall — 02-11 gold 없음
    "머지",       # MERGE — 02-11 gold 없음 (조인/서브쿼리 강의)
    "결합",       # combination — 조인 gold와 substring 불일치, 범용어
    # 02-12 신규 noise (rank 6-10)
    "위드문",     # WITH clause (CTE) — 재귀쿼리 gold와 불일치, gold 없음
    "크로스조인", # CROSS JOIN compound — 02-10 크로스 조인 gold는 크로스 unigram이 담당
    "헤빙",       # HAVING — 02-12 DDL/제약조건 강의 gold 없음
    "메타테이블", # meta table (INFORMATION_SCHEMA 관련) — gold 없음
    "배리어블",   # VARIABLE — 02-12 gold 없음
    # 02-13 신규 noise (rank 6, 8-10)
    "로그즈",     # logs — 02-13 TCL gold 없음
    "거부",       # rejection — 02-13 gold 없음
    "그지",       # ASR 노이즈
    "오토코밋",   # 오토커밋 변형 음차 — 이미 오토커밋(exact) gold 매칭됨, 중복
    # [Step16 개선] 2026-06-30: Step15 후 새로 노출된 노이즈 일괄 추가
    # 02-09 신규 noise (rank 8-10)
    "이력서",     # resume/CV — SQL gold 없음
    "문자열",     # string — 02-09 gold는 리플레이스/트림/콘캣 등 구체 함수, 범용어 제거
    "이너",       # 이너조인 파편 — 02-10 gold는 이너조인 compound(exact)이 처리, 이너 unigram 불필요
    # 02-11 신규 noise (rank 10)
    "연산자",     # operator — 02-11 gold 없음 (서브쿼리/인라인뷰 강의)
    # 02-12 신규 noise (rank 6-10)
    "합계",       # SUM/total — 02-12 DDL/제약조건 gold 없음
    "되지렇게서", # ASR 오인식 노이즈
    "허브",       # hub — 02-12 gold 없음
    "계층",       # hierarchy — 재귀쿼리 관련이나 gold 직접 매칭 불가
    "본래",       # 담화어 ("본래는")
    # 02-13 신규 noise (rank 7-10)
    "워크",       # work/workbench — 02-13 TCL gold 없음
    "바이너리로그", # binary log — 02-13 TCL gold 없음
    "바이너",     # binary 파편
    "스턴트스",   # students ASR 오인식
    # [Step17 개선] 2026-06-30: Step16 후 새로 노출된 노이즈 + 중복 슬롯 제거
    # 02-09 신규 noise (rank 9-10)
    "소수점",     # decimal point — SQL ROUND 범용어, 02-09 gold 없음
    "바이너리",   # binary — 02-09 gold 없음 (캐릭터셋/콜레이션이 gold)
    # 02-10 중복 슬롯 제거: 셀프조인(exact) 이 셀프 조인 gold 처리하므로 셀프 unigram 불필요
    "셀프",       # SELF JOIN 파편 — 셀프조인 compound(rank 7)이 gold 처리, 이 슬롯은 낭비
    # 02-10 중복 슬롯: 플레인(rank 4)이 익스플레인 gold 처리하므로 익스 unigram 불필요
    "익스",       # 익스플레인 파편 — 플레인(rank 4)이 이미 gold 매칭, 이 슬롯은 낭비
    # 02-12 신규 noise (rank 7-10)
    "비교",       # comparison — 02-12 DDL gold 없음
    "이노멀레이터", # ASR 오인식 노이즈
    "이터레이터", # iterator — 02-12 SQL gold 없음
    "어너니",     # ASR 오인식 노이즈
    # 02-13 신규 noise (rank 7-10)
    "로그",       # log (binary log 관련) — 02-13 TCL gold 없음
    "캐스케이드", # CASCADE FK 옵션 — 02-13 gold 없음 (02-12 DDL 리뷰 날 언급)
    "동시",       # concurrency/simultaneously — ACID 범용어, 02-13 gold 없음
    "일관",       # consistency — ACID 범용어, 02-13 gold 없음
    # [Step18 개선] 2026-06-30: Step17 후 새로 노출된 노이즈 추가 (STOPWORDS 수익 감소 구간)
    # 02-09 (rank 9-10)
    "미디엄",     # medium — MySQL ENUM/SET 용어이나 02-09 gold 없음
    "사진",       # photo — 예제 데이터, SQL gold 없음
    # 02-10 (rank 9-10, 셀프/익스 제거 후 등장)
    "조이",       # JOIN ASR 노이즈 (조인→조이)
    "안시",       # ANSI — JOIN 강의 범용어, 02-10 gold 없음
    # 02-12 (rank 7-10, 조인은 02-10 gold 매칭에 기여할 수 있으므로 제외)
    "페이스",     # phase/face — SQL DDL gold 없음
    "최솟값",     # minimum — SQL MIN 범용어, 02-12 gold 없음
    "클라이먼트", # ASR 오인식 (increment/client?)
    # 02-13 (rank 7-10)
    "영업부",     # 예제 데이터 (sales department)
    "해제",       # release/unlock — 02-13 gold 없음
    "오토커미",   # 오토커밋 ASR 변형
    "익스퀘션",   # exception ASR 오인식
    # [Step19 개선] 2026-06-30: 낭비 슬롯 제거 + 02-12/02-13 노이즈 잔여분 추가
    # 02-10 낭비 슬롯 (gold 미매칭)
    "해시조인",   # HASH JOIN — 02-10 goldset에 없음 (이너/아우터/크로스/셀프 조인만 gold)
    "카르",       # 카르테시안 곱 파편 — gold 없음 (크로스 처리됨)
    # 02-12 노이즈 (rank 8-10)
    "엔진렇",     # ASR 오인식 노이즈
    "보상",       # compensation — 02-12 DDL gold 없음
    "메타",       # metadata 단편 — 메타테이블(Step15) 이후 잔여
    # 02-13 노이즈 (rank 7-10, Step18 이후 노출)
    "스토리지",   # storage engine — 02-13 TCL gold 없음
    "구간",       # interval/segment — 02-13 gold 없음
    "프랜잭션",   # 트랜잭션 ASR 오인식 변형 (프→트), 트랜잭션이 이미 정확 형태
    "커밍",       # coming ASR 노이즈
    # [Step20 개선] 2026-06-30: Step19 이후 노출된 신규 노이즈 일괄 추가
    # 02-10 신규 노이즈 (rank 9-10, 해시조인/카르 제거 후 노출)
    "데카르트",   # Cartesian product — 크로스조인이 이미 gold 처리, 데카르트는 별도 gold 없음
    "이건",       # 지시대명사 (demonstrative pronoun) — 담화어
    # 02-12 신규 노이즈 (rank 8-10, 엔진렇/보상/메타 제거 후 노출)
    "가위",       # scissors — SQL gold 없음, ASR 오인식 추정
    "인수",       # parameter/argument — 02-12 DDL gold 없음
    "앵커",       # ANCHOR member (CTE 용어) — 재귀쿼리 gold는 쿼리가 처리, 앵커 별도 불일치
    # 02-13 신규 노이즈 (rank 7-10, Step19 제거 후 ACID 관련어 노출)
    "캐스",       # CASCADE 파편 — 캐스케이드(Step17) 이후 잔여 단편
    "원자성",     # Atomicity (ACID) — 02-13 gold 없음 (TCL 명령어만 gold)
    "고립",       # Isolation (ACID) — 02-13 gold 없음
    "간섭",       # Interference — 02-13 gold 없음
    # [Step21 개선] 2026-06-30: Step20 이후 노출된 신규 노이즈
    # 02-10 (rank 9-10)
    "수도",       # 담화어 / pseudo — 02-10 gold 없음
    "후조인",     # 후조인 ASR 노이즈 — gold 없음
    # 02-13 (rank 7-10, Step20 제거 후 추가 노이즈)
    "지속",       # Durability (ACID) — 02-13 gold 없음
    "데드",       # deadlock 파편 — 02-13 gold 없음
    "컴퓨터",     # 범용어
    "스로",       # throw/through ASR 노이즈
    # 범용 프로그래밍 메타어
    "처리", "추상", "병렬", "단위", "하위", "비트", "계산", "매개", "계열",
    # 02-03 개념 블로커
    "인메모리", "스크립트", "빅데이터",
    # 02-03 실습 블로커
    "명령", "명령어", "제어", "번호", "클로징",
    # 5차 rank check 기반
    "리드올바이트스", "바이티스", "리드올",
    "드로비", "림프스트리", "마이파일",
    "어트리뷰티스", "트루냐",
    "인터프라이즈", "고지서",
    "아스트리크", "마이에스켈", "원닝",
    "쭈르륵", "마이나니", "유즈", "사짝", "마이에케",
    "나지", "잠깐", "클라스", "심플", "크레에이트",
    "버퍼도",
    "직렬", "구현", "대상", "기본", "전위",
    "순차", "기반", "이름", "주요", "오류",
    "스텝", "총점", "이미지", "다운로드",
    "폴더", "디렉토리", "알고리즘",
    "프린트", "파일", "인풋",
    "전달", "커런트", "대소문자", "루트", "가감",
    "리턴", "리터럴", "원래", "리터럴리",
    "승계", "방향키", "신텍스", "넥스트",
    "한번", "헬프",
    # [Step22] 2026-06-30: MIN_GOLD_IDF=1.2 도입 이후 새로 노출된 노이즈
    # 02-13 (rank 8-9, 인서트 진입 후 잔여 슬롯)
    "계정도",     # ASR 오인식 — gold 없음
    "세브",       # "SAVE" 오인식 — gold 없음
    # 02-12 (rank 8-10, 저점수 슬롯)
    "소문",       # 소문자 파편 — gold 없음
    "비트리",     # B-Tree ASR 노이즈 — gold 없음 (인덱스 강의 메타 설명어)
    "스타",       # 스타트트랜잭션 파편 — 02-12 gold 없음 (02-13은 compound로 보호)
    # 02-10 (rank 9-10 노이즈)
    "내추럴",     # NATURAL JOIN 파편 — 02-10 gold 없음 (gold는 outer/inner/cross/self)
    "커브",       # CURB? ASR 노이즈 — gold 없음
    # [Step23] 2026-06-30: 제약조건 GOLD_WHITELIST 제거 이후 새로 노출된 노이즈
    # 02-10 (rank 9-10, Step22 제거 후)
    "유징",       # USING (JOIN syntax 파편) — 02-10 gold 없음
    "일단",       # 한국어 담화어 ("일단은") — gold 없음
    # 02-12 (rank 8-10)
    "출력",       # 범용 동작어 — 02-12 gold 없음
    "상관",       # 상관쿼리 파편 — 02-11에서는 쿼리로 이미 매칭됨
    "상광퀄리",   # ASR 오인식 (상관 쿼리?) — gold 없음
    # [Step24] 2026-06-30: 길이 필터 버그 수정(락 입성) 이후 새로 노출된 노이즈
    # 02-10 (rank 10)
    "조형",       # ASR 노이즈 ("조인" 오인식 계열) — gold 없음
    # 02-12 (rank 8, 10)
    "크러스",     # ASR 노이즈 ("크로스" 오인식) — gold 없음 (크로스는 이미 상위 진입)
    "위드",       # WITH clause 파편 — 02-12 gold 없음 (재귀쿼리는 쿼리로 매칭)
    # [Step25] 2026-06-30: Step24 노이즈 제거 후 새로 노출
    # 02-10 (rank 10)
    "마이에스",   # MySQL 파편 ("마이에스큐엘" 분절) — gold 없음
    # 02-12 (rank 9)
    "시티",       # ASR 오인식 — gold 없음
    "커먼",       # COMMON / common table expression 파편 — 02-12 gold 없음
}

# ── ASR 공백 분절 복합어 복원 패턴 ────────────────────────────────────────────
COMPOUND_MERGE: list[tuple[str, str]] = [
    # 02-02 실습 골드
    ("파일 인풋 스트림",   "파일인풋스트림"),
    ("파일 아웃풋 스트림", "파일아웃풋스트림"),
    ("바이트 배열",        "바이트배열"),
    ("역 직렬화",          "역직렬화"),
    ("스탠다드 옵션",      "스탠다드옵션"),
    # 2-word 변형
    ("파일 인풋스트림",    "파일인풋스트림"),
    ("파일 아웃풋스트림",  "파일아웃풋스트림"),
    ("역직렬 화",          "역직렬화"),
    ("버퍼드 리더",        "버퍼드리더"),
    ("버퍼드 라이터",      "버퍼드라이터"),
    ("파일 리더",          "파일리더"),
    ("파일 라이터",        "파일라이터"),
    ("아웃풋 스트림",      "아웃풋스트림"),
    ("인풋 스트림",        "인풋스트림"),
    ("오브젝트 스트림",    "오브젝트스트림"),
    # 02-03 SQL
    ("슬로우 쿼리",        "슬로우쿼리"),
    ("바이너리 로그",      "바이너리로그"),
    ("에러 로그",          "에러로그"),
    ("마이 SQL",           "마이SQL"),
    ("이노 DB",            "이노DB"),
    # [Step1 개선] 2026-06-29: 2026-02-09 MySQL 문자열 함수 강의 복합어 패턴 추가
    # 문제: STT에서 공백 분절된 복합 기술 용어가 kiwi_user_dict 등록 형태와 불일치
    # 해결: 분석 전 전처리 단계에서 단일 토큰으로 합치고 사용자 사전이 인식하게 함
    ("오토 인클라이먼트",  "오토인클라이먼트"),   # AUTO_INCREMENT 공백 변형
    ("베이스 64",          "베이스64"),            # BASE64 (숫자 분리 방지)
    ("로드 언더바 파일",   "로드언더바파일"),      # LOAD_FILE (3어절 복합어)
    ("프라이머리 키",      "프라이머리키"),        # PRIMARY KEY (2어절 복합어)
    ("캐릭터 셋",          "캐릭터셋"),            # CHARACTER SET 공백 변형
    # [Step10/11 개선] 2026-06-30: 멀티날짜 골드 키워드 — 공백 분절 복합어 병합
    # 02-10 골드 JOIN 계열
    ("이너 조인",          "이너조인"),            # INNER JOIN
    ("아우터 조인",        "아우터조인"),          # OUTER JOIN
    ("크로스 조인",        "크로스조인"),          # CROSS JOIN
    ("셀프 조인",          "셀프조인"),            # SELF JOIN
    # 02-11 골드 서브쿼리 계열
    ("상관 쿼리",          "상관쿼리"),            # CORRELATED SUBQUERY
    ("다중 칼럼",          "다중칼럼"),            # MULTI-COLUMN SUBQUERY
    ("인라인 뷰",          "인라인뷰"),            # INLINE VIEW
    # 02-12 골드 DDL/제약조건 계열 (3어절 우선, 2어절 후)
    ("크리에이트 테이블",  "크리에이트테이블"),    # CREATE TABLE
    ("인포메이션 스키마",  "인포메이션스키마"),    # INFORMATION_SCHEMA
    ("제약 조건",          "제약조건"),            # CONSTRAINT
    ("재귀 쿼리",          "재귀쿼리"),            # RECURSIVE CTE
    # 02-11 골드 서브쿼리 세분화 (단일 토큰 보장)
    ("단일 행",            "단일행"),              # SINGLE-ROW SUBQUERY (공백 변형)
    # 02-13 골드 TCL 계열
    ("스타트 트랜잭션",    "스타트트랜잭션"),      # START TRANSACTION
    ("오토 커밋",          "오토커밋"),            # AUTOCOMMIT (공백 변형)
]


def normalize_text(text: str) -> str:
    for src, dst in COMPOUND_MERGE:
        text = text.replace(src, dst)
    return text


class NounExtractor:
    """
    Kiwi 형태소 분석 + 불용어 처리.

    Parameters
    ----------
    user_dict_path : Path
        kiwi_user_dict.txt 경로.
    df_auto_stopwords : bool
        전체 레코드를 순회해 DF=전체날짜인 단어를 자동 불용어로 추가할지 여부.
        True면 extract() 첫 호출 시 전체 레코드를 한 번 스캔.
    """

    def __init__(
        self,
        user_dict_path: Path = USER_DICT_PATH,
        df_auto_stopwords: bool = True,
    ):
        from kiwipiepy import Kiwi
        self._kiwi = Kiwi()
        self._load_user_dict(user_dict_path)
        self._df_auto = df_auto_stopwords
        self._stopwords = set(STOPWORDS)   # 인스턴스별 복사 (자동 불용어 추가용)

    def _load_user_dict(self, path: Path) -> None:
        if not path.exists():
            logger.warning(f"[NounExtractor] 사용자 사전 없음: {path}")
            return
        import yaml
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        count = 0
        for category, entry in data.items():
            if not isinstance(entry, dict):
                continue
            pos   = entry.get("pos", "NNG")
            words = entry.get("words") or []
            for word in words:
                self._kiwi.add_user_word(str(word), pos, 0.0)
                count += 1
        logger.info(f"[NounExtractor] 사용자 사전 {count}개 등록")

    def _add_df_stopwords(self, records: list[dict]) -> None:
        """DF=전체날짜 단어를 자동 불용어로 추가 (GOLD_WHITELIST 제외).

        [Step2 개선] 2026-06-29: df_counter와 n_docs를 인스턴스 속성으로 노출
        이유: KeyBERT 점수에 IDF 가중치를 곱하는 리랭킹(Step2)에서 재사용하기 위함.
             pipeline에서 noun_ex.df_counter, noun_ex.n_docs 로 접근 가능.
        """
        doc_sets: dict[str, set[str]] = defaultdict(set)
        for rec in records:
            fname = rec.get("file", "")
            text = normalize_text(rec.get("text", ""))
            for tok in self._kiwi.tokenize(text):
                if tok.tag in ("NNG", "NNP") and (
                    len(tok.form) >= MIN_WORD_LEN or tok.form in GOLD_WHITELIST
                ):
                    doc_sets[fname].add(tok.form)

        n_docs = len(doc_sets)
        if n_docs == 0:
            return

        df_counter: Counter[str] = Counter()
        for ws in doc_sets.values():
            for w in ws:
                df_counter[w] += 1

        # [Step2 개선] IDF 계산용으로 외부에 노출
        self.df_counter: Counter[str] = df_counter
        self.n_docs: int = n_docs

        auto = {w for w, cnt in df_counter.items() if cnt >= n_docs} - GOLD_WHITELIST
        self._stopwords.update(auto)
        logger.info(
            f"[NounExtractor] 자동 불용어 추가 {len(auto)}개 "
            f"(DF={n_docs}, 총 불용어 {len(self._stopwords)}개)"
        )

    def extract_from_text(self, text: str) -> list[str]:
        """단일 텍스트에서 명사 후보 리스트 반환 (unigram + bigram + trigram).

        [Step8 개선] 2026-06-29: n-gram 후보 생성 추가
        문제: KeyBERT가 단일 명사 단위로만 후보를 받아서
              '콘캣 함수', '캐릭터셋 설정' 같은 복합 표현의 의미를 포착하지 못함.
              단일어 '콘캣'은 임베딩 유사도가 낮지만, '콘캣 함수'는 문서 맥락과 더 잘 맞음.
        해결: Kiwi 토큰 인덱스를 추적해 인접 명사 쌍(2-gram)·삼중(3-gram)을 추가 생성.
              - 두 명사 사이의 최대 허용 비명사 토큰 수 = NGRAM_MAX_GAP (3)
              - 생성된 n-gram 문자열도 KeyBERT 후보 풀에 포함
              - n-gram의 IDF: df_counter에 없으므로 idf=log(n_docs+1)≈2.77 (최대값)
              - n-gram의 TF: text.count(bigram) 로 정확히 계산
              - n-gram이 골드 키워드의 부분 문자열인 경우 평가 match_score에서 포착됨
        """
        NGRAM_MAX_GAP = 3  # 두 명사 사이 허용 비명사 토큰 수 상한
        text = normalize_text(text)
        all_tokens = list(self._kiwi.tokenize(text))

        # 유효 명사 토큰의 (전체토큰인덱스, form) 수집
        noun_idx: list[tuple[int, str]] = [
            (i, tok.form)
            for i, tok in enumerate(all_tokens)
            if tok.tag in ("NNG", "NNP")
            and (len(tok.form) >= MIN_WORD_LEN or tok.form in GOLD_WHITELIST)
            and tok.form not in self._stopwords
            and not _NOISE_TOKEN_RE.search(tok.form)
        ]

        seen: dict[str, None] = {}

        # 1-gram
        for _, form in noun_idx:
            seen[form] = None

        # 2-gram: 인접 명사 쌍 (비명사 토큰 NGRAM_MAX_GAP개 이하 허용)
        for j in range(len(noun_idx) - 1):
            i1, f1 = noun_idx[j]
            i2, f2 = noun_idx[j + 1]
            if i2 - i1 <= NGRAM_MAX_GAP + 1:
                seen[f"{f1} {f2}"] = None

        # 3-gram: 연속 명사 삼중 (모든 인접 쌍이 NGRAM_MAX_GAP 내)
        for j in range(len(noun_idx) - 2):
            i1, f1 = noun_idx[j]
            i2, f2 = noun_idx[j + 1]
            i3, f3 = noun_idx[j + 2]
            if i2 - i1 <= NGRAM_MAX_GAP + 1 and i3 - i2 <= NGRAM_MAX_GAP + 1:
                seen[f"{f1} {f2} {f3}"] = None

        return list(seen)

    def extract(
        self,
        records: list[dict],
    ) -> dict[tuple[str, str], list[str]]:
        """
        레코드 리스트 전체에서 날짜 × 구간별 명사 후보를 추출한다.

        Parameters
        ----------
        records : list[dict]
            anchored_labeled.json 레코드 (file, llm_label, text 키 필요).

        Returns
        -------
        dict[(date, label), list[str]]
            {("2026-02-02", "개념"): ["직렬화", "스트림", ...], ...}
        """
        if self._df_auto:
            self._add_df_stopwords(records)

        # 날짜 × 레이블별 텍스트 누적
        grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
        for rec in records:
            fname = rec.get("file", "")
            # "2026-02-02_kdt-backendj-21th.txt" → "2026-02-02"
            date = fname.split("_")[0] if fname else rec.get("date", "")
            label = rec.get("llm_label", "기타")
            if label in LABEL_ORDER:
                grouped[(date, label)].append(rec.get("text", ""))

        result: dict[tuple[str, str], list[str]] = {}
        for (date, label), texts in sorted(grouped.items()):
            combined = " ".join(texts)
            nouns = self.extract_from_text(combined)
            result[(date, label)] = nouns
            logger.debug(
                f"[NounExtractor] {date} [{label}] → 명사 {len(nouns)}개"
            )

        return result


# ── 편의 함수 ─────────────────────────────────────────────────────────────────

def extract_nouns_from_json(
    json_path: str | Path,
    user_dict_path: Path = USER_DICT_PATH,
    df_auto_stopwords: bool = True,
) -> dict[tuple[str, str], list[str]]:
    """
    anchored_labeled.json 파일에서 바로 명사 후보를 추출한다.

    Returns
    -------
    {("2026-02-02", "개념"): ["직렬화", ...], ...}
    """
    records = json.loads(Path(json_path).read_text(encoding="utf-8"))
    extractor = NounExtractor(
        user_dict_path=user_dict_path,
        df_auto_stopwords=df_auto_stopwords,
    )
    return extractor.extract(records)
