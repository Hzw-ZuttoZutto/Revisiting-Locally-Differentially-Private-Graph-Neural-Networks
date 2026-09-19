from pathlib import Path

import scripts.aec.fixed_runner as fixed_runner


def test_fixed_batch_builds_one_job_per_point(tmp_path, monkeypatch):
    monkeypatch.setenv("AEC_GPU_IDS", "0,1")
    monkeypatch.setenv("AEC_MAX_PARALLEL_PER_GPU", "2")
    points = fixed_runner._coverage(1, fixed_runner.fixed_points(1))[:2]

    batch, configs = fixed_runner._build_fixed_batch(
        points,
        root=tmp_path,
        repeats=3,
    )

    assert batch is not None
    assert len(configs) == 2
    assert len(batch.jobs) == 2
    assert len({job.job_id for job in batch.jobs}) == 2
    assert len({str(job.job_dir) for job in batch.jobs}) == 2
    assert batch.execution.worker_ids == [0, 1]
    assert batch.execution.max_parallel_per_worker == 2


def test_run_fixed_invokes_one_global_batch(monkeypatch, tmp_path):
    monkeypatch.setenv("AEC_GPU_IDS", "0,1")
    monkeypatch.setenv("AEC_MAX_PARALLEL_PER_GPU", "2")
    monkeypatch.setattr(fixed_runner, "WORK_ROOT", tmp_path)
    monkeypatch.setattr(fixed_runner, "_aggregate_search_output", lambda *args: None)
    calls = []

    def fake_run_batch(batch, *, repo_root):
        calls.append((batch, repo_root))
        return len(batch.jobs), 0, 0, batch.output_root / "manifest.csv"

    monkeypatch.setattr(fixed_runner.core, "run_batch_search", fake_run_batch)
    jobs = fixed_runner.run_fixed(1, repeats=3, limit=2, execute=True)

    assert len(jobs) == 2
    assert len(calls) == 1
    assert len(calls[0][0].jobs) == 2


def test_table_planning_uses_the_same_batch_path(monkeypatch, tmp_path):
    monkeypatch.setattr(fixed_runner, "WORK_ROOT", tmp_path)
    calls = []
    monkeypatch.setattr(
        fixed_runner.core,
        "run_batch_search",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    summary = fixed_runner.run_fixed_table("table4", repeats=3, execute=False)

    assert summary["table"] == "table4"
    assert len(summary["jobs"]) > 1
    assert calls == []


def test_core_scheduler_receives_all_fixed_jobs(monkeypatch, tmp_path):
    monkeypatch.setenv("AEC_GPU_IDS", "0,1")
    monkeypatch.setenv("AEC_MAX_PARALLEL_PER_GPU", "2")
    points = fixed_runner._coverage(1, fixed_runner.fixed_points(1))[:2]
    batch, _ = fixed_runner._build_fixed_batch(points, root=tmp_path, repeats=3)
    created = []

    class FakeScheduler:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.tasks = []
            created.append(self)

        def enqueue(self, task):
            self.tasks.append(task)

        def run(self, on_result):
            return None

    monkeypatch.setattr(fixed_runner.core, "MultiPoolScheduler", FakeScheduler)
    fixed_runner.core.run_batch_search(batch, repo_root=fixed_runner.REPO_ROOT)

    assert len(created) == 1
    assert created[0].kwargs["worker_ids"] == [0, 1]
    assert created[0].kwargs["max_parallel_per_worker"] == 2
    assert {task.combo_key for task in created[0].tasks} == {
        job.job_id for job in batch.jobs
    }
