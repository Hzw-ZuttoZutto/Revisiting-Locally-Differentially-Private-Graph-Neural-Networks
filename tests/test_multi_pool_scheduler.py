from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from hparams_search_scripts.multi_pool_scheduler import MultiPoolScheduler, SchedulerTask, TaskResult


class MultiPoolSchedulerTests(unittest.TestCase):
    def _task(
        self,
        *,
        task_id: str,
        combo_key: str,
        pool: str,
        cwd: Path,
        logs_dir: Path,
        command: list[str] | None = None,
    ) -> SchedulerTask:
        if command is None:
            command = [sys.executable, "-c", "print('ok')"]
        return SchedulerTask(
            task_id=task_id,
            combo_key=combo_key,
            pool=pool,
            command=command,
            env={},
            cwd=cwd,
            log_path=logs_dir / f"{task_id}.log",
            retry_count=0,
            label=task_id,
            log_index_path=logs_dir / f"{combo_key}_index.csv",
        )

    def test_round_robin_between_pools(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scheduler = MultiPoolScheduler(
                worker_ids=[0],
                max_parallel_per_worker=1,
                poll_interval_sec=0.01,
                launch_interval_sec=0.0,
                device="cpu",
            )
            scheduler.enqueue(self._task(task_id="g1", combo_key="c1", pool="grid", cwd=root, logs_dir=root))
            scheduler.enqueue(self._task(task_id="g2", combo_key="c1", pool="grid", cwd=root, logs_dir=root))
            scheduler.enqueue(self._task(task_id="v1", combo_key="c2", pool="verify", cwd=root, logs_dir=root))
            scheduler.enqueue(self._task(task_id="v2", combo_key="c2", pool="verify", cwd=root, logs_dir=root))

            finished: list[str] = []

            def on_result(result: TaskResult):
                finished.append(result.task.task_id)
                return []

            scheduler.run(on_result)
            self.assertEqual(finished, ["g1", "v1", "g2", "v2"])

    def test_dynamic_enqueue_after_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scheduler = MultiPoolScheduler(
                worker_ids=[0],
                max_parallel_per_worker=1,
                poll_interval_sec=0.01,
                launch_interval_sec=0.0,
                device="cpu",
            )
            scheduler.enqueue(self._task(task_id="grid1", combo_key="comboA", pool="grid", cwd=root, logs_dir=root))

            finished: list[str] = []

            def on_result(result: TaskResult):
                finished.append(result.task.task_id)
                if result.task.task_id == "grid1":
                    return [
                        self._task(
                            task_id="verify1",
                            combo_key="comboA",
                            pool="verify",
                            cwd=root,
                            logs_dir=root,
                        )
                    ]
                return []

            scheduler.run(on_result)
            self.assertEqual(finished, ["grid1", "verify1"])

    def test_failed_combo_is_cancelled_while_other_combo_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scheduler = MultiPoolScheduler(
                worker_ids=[0],
                max_parallel_per_worker=1,
                poll_interval_sec=0.01,
                launch_interval_sec=0.0,
                device="cpu",
            )
            scheduler.enqueue(
                self._task(
                    task_id="c1_fail",
                    combo_key="combo1",
                    pool="grid",
                    cwd=root,
                    logs_dir=root,
                    command=[sys.executable, "-c", "import sys; sys.exit(2)"],
                )
            )
            scheduler.enqueue(self._task(task_id="c1_skip", combo_key="combo1", pool="grid", cwd=root, logs_dir=root))
            scheduler.enqueue(self._task(task_id="c2_ok", combo_key="combo2", pool="grid", cwd=root, logs_dir=root))

            finished: list[str] = []
            failed: list[str] = []

            def on_result(result: TaskResult):
                finished.append(result.task.task_id)
                if not result.success:
                    failed.append(result.task.task_id)
                    scheduler.cancel_combo(result.task.combo_key)
                return []

            scheduler.run(on_result)
            self.assertEqual(failed, ["c1_fail"])
            self.assertIn("c2_ok", finished)
            self.assertNotIn("c1_skip", finished)


if __name__ == "__main__":
    unittest.main()

