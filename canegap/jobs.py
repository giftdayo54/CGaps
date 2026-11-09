"""In-process background job runner (thread pool) for analyses.

Design
  * A job is recorded in the database first (status 'queued'); workers open their OWN SQLite connection.
  * Progress, current field and a cancel flag live in the database, so any Streamlit session can show/cancel jobs.
  * Cancellation is cooperative and checked between fields; fields completed so far are kept.
  * Identical jobs that are still queued/running are refused (see pipeline.queue_analysis).
  * Jobs run inside the app process. If the process restarts, queued/running jobs are marked 'interrupted' at startup.
    For multi-server or long-running production use, replace this with a real queue (Celery/RQ) using the same
    queue_analysis/execute_job functions.
"""
from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor

from . import db
from .pipeline import execute_job


class JobRunner:
    def __init__(self, db_path, max_workers: int = 2):
        self.db_path = db_path
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="canegap-job")
        self._futures: dict[int, Future] = {}
        self._lock = threading.Lock()
        c = db.connect(db_path)
        try:
            self.recovered = db.recover_orphans(c)
        finally:
            c.close()

    def _work(self, job_pk, provider, model):
        conn = db.connect(self.db_path)
        try:
            execute_job(conn, job_pk, provider, model, should_cancel=lambda: db.cancel_requested(conn, job_pk))
        except Exception:  # noqa: BLE001  (execute_job already recorded the failure on the job)
            pass
        finally:
            conn.close()

    def submit(self, job_pk: int, provider, model=None) -> Future:
        with self._lock:
            fut = self._pool.submit(self._work, job_pk, provider, model)
            self._futures[job_pk] = fut
            return fut

    def is_active(self, job_pk: int) -> bool:
        f = self._futures.get(job_pk)
        return bool(f and not f.done())

    def wait(self, job_pk: int, timeout: float | None = None) -> None:
        f = self._futures.get(job_pk)
        if f:
            f.result(timeout)

    def shutdown(self):
        self._pool.shutdown(wait=False, cancel_futures=True)
