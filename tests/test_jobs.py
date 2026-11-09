import threading

import pytest

from canegap import db, pipeline, samples
from canegap.geojson_import import validate_geojson
from canegap.jobs import JobRunner
from canegap.pipeline import DuplicateJob, Params, queue_analysis
from canegap.providers import DemoProvider

FIELDS = ["F01", "F02", "F05", "F06"]


@pytest.fixture()
def env(tmp_path):
    path = tmp_path / "jobs.sqlite"
    conn = db.connect(path)
    ids = db.save_import(conn, validate_geojson(samples.farm_with_fields(True), outline_rows=[6]), "t")[0]
    runner = JobRunner(path, max_workers=2)
    yield conn, ids, runner, path
    runner.shutdown()


def _queue(conn, ids, prov=None, fields=FIELDS, **kw):
    return queue_analysis(conn, ids["farm_pk"], ids["version_pk"], fields, prov or DemoProvider(), "2026-08-01", "2026-10-15",
                          created_by="tester", **kw)


def test_background_job_runs_to_completion_with_progress(env):
    conn, ids, runner, _ = env
    job = _queue(conn, ids)
    assert db.get_job(conn, job)["status"] == "queued" and db.get_job(conn, job)["progress_total"] == 4
    runner.submit(job, DemoProvider())
    runner.wait(job, 60)
    j = db.get_job(conn, job)
    assert j["status"] == "completed" and j["progress_done"] == 4 and j["created_by"] == "tester"
    assert len(db.job_results(conn, job)) == 4 and len(db.job_gaps(conn, job)) > 0


def test_background_result_equals_synchronous_result(env):
    conn, ids, runner, _ = env
    sync = pipeline.run_analysis(conn, ids["farm_pk"], ids["version_pk"], FIELDS, DemoProvider(), "2026-08-01", "2026-10-15")
    job = _queue(conn, ids)
    runner.submit(job, DemoProvider()); runner.wait(job, 60)
    a = db.job_results(conn, sync).set_index("field_id")[["status", "gap_area_ha", "n_gaps"]]
    b = db.job_results(conn, job).set_index("field_id")[["status", "gap_area_ha", "n_gaps"]]
    assert a.equals(b)


def test_identical_active_job_is_refused_but_allowed_after_finish(env):
    conn, ids, runner, _ = env
    j1 = _queue(conn, ids)
    with pytest.raises(DuplicateJob, match=f"#{j1}"):
        _queue(conn, ids)
    p = Params(); p.edge_buffer_m = 20.0
    assert _queue(conn, ids, params=p)                                  # different parameters = different job
    runner.submit(j1, DemoProvider()); runner.wait(j1, 60)
    assert _queue(conn, ids)                                            # finished -> a re-run is legitimate


def test_cancel_stops_between_fields_and_keeps_completed_fields(env):
    conn, ids, runner, _ = env
    started, release = threading.Event(), threading.Event()

    class Gate(DemoProvider):
        def get_composite(self, *a, **k):
            started.set(); release.wait(30)
            return super().get_composite(*a, **k)

    job = _queue(conn, ids, Gate())
    runner.submit(job, Gate())
    assert started.wait(30)
    assert db.request_cancel(conn, job)
    release.set()
    runner.wait(job, 60)
    j = db.get_job(conn, job)
    assert j["status"] == "cancelled" and "Cancelled by user" in j["error"]
    assert len(db.job_results(conn, job)) == 1                          # first field finished, rest skipped
    assert not db.request_cancel(conn, job)                             # cannot cancel a finished job


def test_failed_job_records_error_and_keeps_earlier_fields(env, monkeypatch):
    conn, ids, runner, _ = env
    real = pipeline.analyse_field
    def boom(field_id, *a, **k):
        if field_id == "F05":
            raise RuntimeError("bug in field F05")
        return real(field_id, *a, **k)
    monkeypatch.setattr(pipeline, "analyse_field", boom)
    job = _queue(conn, ids)
    runner.submit(job, DemoProvider()); runner.wait(job, 60)
    j = db.get_job(conn, job)
    assert j["status"] == "failed" and "bug in field F05" in j["error"]
    assert {r.field_id for r in db.job_results(conn, job).itertuples()} == {"F01", "F02"}


def test_orphaned_jobs_are_marked_interrupted_on_startup(env, tmp_path):
    conn, ids, runner, path = env
    j1, j2 = _queue(conn, ids), _queue(conn, ids, params=Params(edge_buffer_m=20.0))
    db.set_job_running(conn, j1)
    done = pipeline.run_analysis(conn, ids["farm_pk"], ids["version_pk"], ["F01"], DemoProvider(), "2026-08-01", "2026-10-15")
    r2 = JobRunner(path)                                                # simulates an application restart
    assert r2.recovered == 2
    assert db.get_job(conn, j1)["status"] == "interrupted" and db.get_job(conn, j2)["status"] == "interrupted"
    assert db.get_job(conn, done)["status"] == "completed"
    assert "restarted" in db.get_job(conn, j1)["error"]
    r2.shutdown()


def test_queue_validations(env):
    conn, ids, runner, _ = env
    with pytest.raises(ValueError, match="No valid fields"):
        _queue(conn, ids, fields=["NOPE"])


def test_old_database_gets_new_job_columns(tmp_path):
    import sqlite3
    path = tmp_path / "old.sqlite"
    c = sqlite3.connect(path)
    c.execute("CREATE TABLE jobs(id INTEGER PRIMARY KEY, farm_pk INTEGER, status TEXT)")
    c.execute("INSERT INTO jobs(farm_pk,status) VALUES(1,'completed')"); c.commit(); c.close()
    conn = db.connect(path)
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(jobs)")}
    assert {"progress_done", "cancel_requested", "fingerprint", "created_by"} <= cols
