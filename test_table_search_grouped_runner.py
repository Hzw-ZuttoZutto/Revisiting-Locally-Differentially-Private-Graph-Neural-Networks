from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from hparams_search_scripts import mechanism_stage_utils
from hparams_search_scripts import run_mechanism_hparam_search as search_runner
from hparams_search_scripts import table_search_suite_core as core


REPO_ROOT = Path(__file__).resolve().parent


def _load_batch_spec(relative_config_path: str) -> core.BatchSpec:
    config_path = REPO_ROOT / relative_config_path
    search_config = search_runner.load_search_config(config_path)
    return search_runner.build_batch_spec(
        search_config=search_config,
        output_root=REPO_ROOT / ".grouped-runner-test-output-unused",
        config_copy_source=config_path,
    )


def _candidate_ids_from_command(command: list[str]) -> list[int]:
    return [
        int(command[index + 1])
        for index, token in enumerate(command)
        if token == "--candidate_id"
    ]


class _RecordingScheduler:
    instances: list["_RecordingScheduler"] = []

    def __init__(self, **_kwargs) -> None:
        self.tasks: list[core.SchedulerTask] = []
        self.__class__.instances.append(self)

    def enqueue(self, task: core.SchedulerTask) -> None:
        self.tasks.append(task)

    def cancel_combo(self, _combo_key: str) -> None:
        return None

    def run(self, _on_result) -> None:
        return None


class GroupedRunnerSelectionTest(unittest.TestCase):
    def test_all_rebuttal_heter_nfr_configs_use_materialized_runner(self) -> None:
        figure1_configs = sorted(
            path.relative_to(REPO_ROOT).as_posix()
            for path in (REPO_ROOT / "config_rebuttal" / "figure1_heter").glob(
                "*/*/figure3_pipeline[34].yaml"
            )
        )
        figure6_configs = sorted(
            path.relative_to(REPO_ROOT).as_posix()
            for path in (REPO_ROOT / "config_rebuttal" / "figure6_heter").glob("*.yaml")
        )
        self.assertEqual(len(figure1_configs), 12)
        self.assertEqual(len(figure6_configs), 16)

        for config_path in figure1_configs + figure6_configs:
            with self.subTest(config=config_path):
                self.assertEqual(core._grouped_runner_mode(_load_batch_spec(config_path)), "materialized")

    def test_real_candidate_spaces_group_by_x_steps_and_tao2(self) -> None:
        cases = (
            (
                "config_rebuttal/figure1_heter/attributedgraph-flickr/gat/figure3_pipeline3.yaml",
                1260,
                35,
                36,
            ),
            (
                "config_rebuttal/figure1_heter/attributedgraph-flickr/gat/figure3_pipeline4.yaml",
                1260,
                35,
                36,
            ),
            ("config_rebuttal/figure6_heter/tao2=0.5.yaml", 252, 7, 36),
        )

        for config_path, expected_candidates, expected_groups, expected_group_size in cases:
            with self.subTest(config=config_path):
                spec = _load_batch_spec(config_path)
                candidates = mechanism_stage_utils.job_candidates(spec.jobs[0].job_spec)
                groups = core._group_grid_candidates(candidates)

                self.assertEqual(len(candidates), expected_candidates)
                self.assertEqual(len(groups), expected_groups)
                self.assertEqual({len(group) for group in groups}, {expected_group_size})
                self.assertEqual(
                    sorted(candidate.candidate_id for group in groups for candidate in group),
                    sorted(candidate.candidate_id for candidate in candidates),
                )
                for group in groups:
                    self.assertEqual(
                        {
                            (
                                candidate.x_steps,
                                mechanism_stage_utils.canonical_optional_float_text(candidate.tao2),
                            )
                            for candidate in group
                        },
                        {
                            (
                                group[0].x_steps,
                                mechanism_stage_utils.canonical_optional_float_text(group[0].tao2),
                            )
                        },
                    )

    def test_raw_nfr_semantic_gate_rejects_unsupported_state(self) -> None:
        fixed_params = {
            "feature": "raw",
            "use_nfr": True,
            "mechanism": "mbm",
            "backbone": "gat",
            "smoother": "hoa",
        }
        for mechanism in ("mbm", "pm", "hds"):
            with self.subTest(mechanism=mechanism):
                candidate = {**fixed_params, "mechanism": mechanism}
                self.assertTrue(core._uses_semantic_raw_grouped_state_runner(candidate))

        invalid_variants = (
            {**fixed_params, "mechanism": "lpm"},
            {**fixed_params, "feature": "random_normal"},
            {**fixed_params, "backbone": "unsupported"},
            {**fixed_params, "smoother": "unsupported"},
        )
        for candidate in invalid_variants:
            with self.subTest(candidate=candidate):
                self.assertFalse(core._uses_semantic_raw_grouped_state_runner(candidate))

        job_spec = copy.deepcopy(
            _load_batch_spec(
                "config_rebuttal/figure1_heter/attributedgraph-flickr/gat/figure3_pipeline4.yaml"
            ).jobs[0].job_spec
        )
        job_spec["fixed_params"]["mechanism"] = "lpm"
        legacy_allowlisted_path = REPO_ROOT / "main_add" / "gat" / "config.yaml"
        self.assertIsNone(core._job_grouped_runner_mode(legacy_allowlisted_path, job_spec))

    def test_mixed_job_capabilities_disable_grouped_runner(self) -> None:
        spec = _load_batch_spec(
            "config_rebuttal/figure1_heter/attributedgraph-flickr/gat/figure3_pipeline4.yaml"
        )
        eligible_job = spec.jobs[0]
        ineligible_job_spec = copy.deepcopy(spec.jobs[1].job_spec)
        ineligible_job_spec["fixed_params"]["mechanism"] = "lpm"
        ineligible_job = core.BatchJob(
            job_id="ineligible-job",
            job_dir=spec.jobs[1].job_dir,
            display_name="ineligible job",
            job_spec=ineligible_job_spec,
        )
        mixed_spec = core.BatchSpec(
            output_root=spec.output_root,
            execution=spec.execution,
            jobs=[eligible_job, ineligible_job],
            config_copy_source=spec.config_copy_source,
        )
        self.assertIsNone(core._grouped_runner_mode(mixed_spec))

    def test_sim_epoch_refresh_remains_outside_materialized_runner(self) -> None:
        spec = _load_batch_spec(
            "config_rebuttal/figure1_heter/attributedgraph-flickr/gat/figure3_pipeline4.yaml"
        )
        job_spec = copy.deepcopy(spec.jobs[0].job_spec)
        job_spec["fixed_params"].update(
            {
                "feature": "sim",
                "use_nfr": False,
                "mechanism": "mbm",
            }
        )
        job_spec["defaults"]["trainer"]["sim_epoch_refresh"] = True
        self.assertIsNone(
            core._job_grouped_runner_mode(REPO_ROOT / "arbitrary" / "config.yaml", job_spec)
        )

    def test_run_batch_search_enqueues_group_commands(self) -> None:
        spec = _load_batch_spec(
            "config_rebuttal/figure1_heter/attributedgraph-flickr/gat/figure3_pipeline3.yaml"
        )
        source_job = spec.jobs[0]
        candidates_by_id = {
            candidate.candidate_id: candidate
            for candidate in mechanism_stage_utils.job_candidates(source_job.job_spec)
        }
        selected_ids = [1, 6, 2, 7]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir) / "output"
            job_spec = copy.deepcopy(source_job.job_spec)
            job_spec["candidates"] = [candidates_by_id[candidate_id].to_dict() for candidate_id in selected_ids]
            job = core.BatchJob(
                job_id=source_job.job_id,
                job_dir=output_root / "job",
                display_name=source_job.display_name,
                job_spec=job_spec,
            )
            tiny_spec = core.BatchSpec(
                output_root=output_root,
                execution=spec.execution,
                jobs=[job],
                config_copy_source=spec.config_copy_source,
            )

            _RecordingScheduler.instances.clear()
            with mock.patch.object(core, "MultiPoolScheduler", _RecordingScheduler):
                core.run_batch_search(tiny_spec, repo_root=REPO_ROOT)

            tasks = _RecordingScheduler.instances[0].tasks
            self.assertEqual(len(tasks), 2)
            self.assertEqual(sum(task.work_units for task in tasks), len(selected_ids))
            self.assertTrue(all(task.task_id.startswith("grid_group_") for task in tasks))
            self.assertTrue(
                all(Path(task.command[1]).name == "run_mechanism_group_task.py" for task in tasks)
            )
            self.assertTrue(all(task.command[3:5] == ["--stage", "grid"] for task in tasks))
            self.assertEqual(
                sorted(candidate_id for task in tasks for candidate_id in _candidate_ids_from_command(task.command)),
                sorted(selected_ids),
            )
            for task in tasks:
                command_candidates = [candidates_by_id[candidate_id] for candidate_id in _candidate_ids_from_command(task.command)]
                self.assertEqual(
                    {
                        (
                            candidate.x_steps,
                            mechanism_stage_utils.canonical_optional_float_text(candidate.tao2),
                        )
                        for candidate in command_candidates
                    },
                    {
                        (
                            command_candidates[0].x_steps,
                            mechanism_stage_utils.canonical_optional_float_text(command_candidates[0].tao2),
                        )
                    },
                )


if __name__ == "__main__":
    unittest.main()
