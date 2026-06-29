"""분석 작업(job) 진행 상태 저장소 + 진행률 리포터.

비동기 분석을 백그라운드로 돌리고, 프론트가 ``GET /api/v1/analysis/job/{id}`` 로
폴링한다. 진행 신호는 ``ProgressSink`` 를 통해 파이프라인에서 보고된다.
(LLM 호출을 추가하지 않으므로 엔진 비용 영향 없음.)

스레드 안전: ``run`` 은 ``asyncio.to_thread`` 로 워커 스레드에서 돌고, 폴링 핸들러는
이벤트 루프 스레드에서 읽으므로 모든 변경/조회를 ``threading.Lock`` 으로 보호한다.
저장소는 in-memory(프로세스 단일) — 단일 워커 개발 환경 기준.
"""
from __future__ import annotations

import threading
import uuid
from datetime import datetime
from typing import Any, Callable

_lock = threading.Lock()
_jobs: dict[str, "Job"] = {}
_MAX_LOGS = 200


def _now_hms() -> str:
    return datetime.now().strftime("%H:%M:%S")


class Job:
    """단일 분석 작업의 진행 상태."""

    def __init__(self, job_id: str, label: str = "") -> None:
        self.id = job_id
        self.label = label
        self.status = "pending"  # pending | running | done | error
        self.percent = 0.0
        self.stage = "대기 중"
        self.completed = 0
        self.total = 0
        self.logs: list[dict[str, str]] = []
        self.result: dict[str, Any] | None = None
        self.error: str | None = None
        self.created_at = datetime.now().isoformat(timespec="seconds")

    def snapshot(self) -> dict[str, Any]:
        return {
            "job_id": self.id,
            "label": self.label,
            "status": self.status,
            "percent": round(self.percent, 1),
            "stage": self.stage,
            "completed": self.completed,
            "total": self.total,
            "logs": list(self.logs),
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
        }


def _append_log(job: Job, message: str, level: str) -> None:
    job.logs.append({"ts": _now_hms(), "level": level, "message": message})
    if len(job.logs) > _MAX_LOGS:
        del job.logs[: len(job.logs) - _MAX_LOGS]


def create_job(label: str = "") -> str:
    job_id = uuid.uuid4().hex[:12]
    with _lock:
        _jobs[job_id] = Job(job_id, label)
    return job_id


def get_job(job_id: str) -> dict[str, Any] | None:
    with _lock:
        job = _jobs.get(job_id)
        return job.snapshot() if job else None


def mark_running(job_id: str) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job:
            job.status = "running"
            _append_log(job, "분석 시작", "info")


def finish_job(job_id: str, result: dict[str, Any]) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job:
            job.status = "done"
            job.percent = 100.0
            job.stage = "완료"
            job.result = result
            _append_log(job, "분석 완료", "success")


def fail_job(job_id: str, error: str) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job:
            job.status = "error"
            job.error = error
            job.stage = "오류"
            _append_log(job, f"오류: {error}", "error")


# ── 파이프라인 → 진행률 보고 인터페이스 ──────────────────────────────
class ProgressSink:
    """기본 구현은 모두 무시(no-op). 파이프라인은 이 인터페이스에만 의존한다."""

    def stage(self, message: str, percent: float | None = None) -> None: ...
    def items(self, total: int) -> None: ...
    def item_done(self, key: str, score: Any, completed: int, total: int) -> None: ...
    def log(self, message: str, level: str = "info") -> None: ...


class NullProgressSink(ProgressSink):
    pass


NULL_SINK = NullProgressSink()


class JobProgressSink(ProgressSink):
    """진행 신호를 job 저장소에 기록한다. 항목 단위 완료를 진행률로 환산.

    진행률 모델: 전처리 8% → 라벨링 16% → 항목 채점 16~92% → 스코어카드 96% → 완료 100%.
    """

    def __init__(self, job_id: str) -> None:
        self.job_id = job_id

    def _mutate(self, fn: Callable[[Job], None]) -> None:
        with _lock:
            job = _jobs.get(self.job_id)
            if job is not None:
                fn(job)

    def stage(self, message: str, percent: float | None = None) -> None:
        def fn(job: Job) -> None:
            job.status = "running"
            job.stage = message
            if percent is not None:
                job.percent = max(job.percent, percent)
            _append_log(job, message, "info")

        self._mutate(fn)

    def items(self, total: int) -> None:
        self._mutate(lambda job: setattr(job, "total", total))

    def item_done(self, key: str, score: Any, completed: int, total: int) -> None:
        def fn(job: Job) -> None:
            job.completed = completed
            job.total = total
            job.percent = max(job.percent, 16 + (completed / max(total, 1)) * 76)
            _append_log(job, f"[{completed}/{total}] {key} 채점 완료 (점수 {score})", "info")

        self._mutate(fn)

    def log(self, message: str, level: str = "info") -> None:
        self._mutate(lambda job: _append_log(job, message, level))
