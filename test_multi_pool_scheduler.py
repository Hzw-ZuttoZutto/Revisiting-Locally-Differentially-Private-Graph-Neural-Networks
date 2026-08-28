from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock

from hparams_search_scripts.multi_pool_scheduler import MultiPoolScheduler, SchedulerTask


def _task() -> SchedulerTask:
    return SchedulerTask(
        task_id="task",
        combo_key="combo",
        pool="grid",
        command=["true"],
        env={},
        cwd=Path.cwd(),
        log_path=Path("unused.log"),
        retry_count=0,
        label="test task",
    )


def _choose_worker(
    scheduler: MultiPoolScheduler,
    counts: dict[int, int],
) -> int | None:
    return scheduler._choose_worker_for_task(
        _task(),
        running_count_by_worker=counts,
        running_count_by_group_worker={},
    )


class TemporaryGpu2CapTest(unittest.TestCase):
    def _scheduler(self, *, max_parallel: int = 20) -> MultiPoolScheduler:
        return MultiPoolScheduler(
            worker_ids=[0, 1, 2, 3],
            max_parallel_per_worker=max_parallel,
            poll_interval_sec=0.1,
            launch_interval_sec=0.0,
            device="gpu",
        )

    def test_flag_caps_only_gpu2_at_15_and_keeps_it_dynamic_below_cap(self) -> None:
        with mock.patch.dict(os.environ, {"REBUTTAL_TEMP_GPU2_CAP15": "1"}):
            scheduler = self._scheduler()

        self.assertEqual(_choose_worker(scheduler, {0: 20, 1: 20, 2: 14, 3: 20}), 2)
        self.assertIsNone(_choose_worker(scheduler, {0: 20, 1: 20, 2: 15, 3: 20}))
        self.assertEqual(_choose_worker(scheduler, {0: 19, 1: 20, 2: 15, 3: 20}), 0)

    def test_flag_is_off_by_default(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            scheduler = self._scheduler()

        self.assertEqual(_choose_worker(scheduler, {0: 20, 1: 20, 2: 15, 3: 20}), 2)

    def test_gpu2_cap_never_increases_a_lower_global_limit(self) -> None:
        with mock.patch.dict(os.environ, {"REBUTTAL_TEMP_GPU2_CAP15": "true"}):
            scheduler = self._scheduler(max_parallel=10)

        self.assertIsNone(_choose_worker(scheduler, {0: 10, 1: 10, 2: 10, 3: 10}))

    def test_invalid_flag_value_fails_fast(self) -> None:
        with mock.patch.dict(os.environ, {"REBUTTAL_TEMP_GPU2_CAP15": "maybe"}):
            with self.assertRaisesRegex(ValueError, "REBUTTAL_TEMP_GPU2_CAP15"):
                self._scheduler()


if __name__ == "__main__":
    unittest.main()
