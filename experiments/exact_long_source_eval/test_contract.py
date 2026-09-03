from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from contract import (
    apply_nominal_phase_zero_contract,
    classify_episodes,
    reset_episode_in_inference_mode,
)


def _fake_env_cfg() -> SimpleNamespace:
    motion = SimpleNamespace(
        motion_file="old.npz",
        min_sample_idx=12,
        max_sample_idx=99,
        pose_range={
            "x": (-0.02, 0.02),
            "y": (-0.02, 0.02),
            "z": (-0.005, 0.005),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.1, 0.1),
        },
        velocity_range={
            "x": (-0.25, 0.25),
            "y": (-0.25, 0.25),
            "z": (-0.1, 0.1),
            "roll": (-0.26, 0.26),
            "pitch": (-0.26, 0.26),
            "yaw": (-0.39, 0.39),
        },
        joint_position_range=(-0.05, 0.05),
        debug_vis=True,
    )
    return SimpleNamespace(
        scene=SimpleNamespace(num_envs=4096),
        commands=SimpleNamespace(motion=motion),
        observations=SimpleNamespace(policy=SimpleNamespace(enable_corruption=True)),
        events=SimpleNamespace(physics_material=object(), push_robot=object()),
        curriculum=SimpleNamespace(adr=object()),
        terminations=SimpleNamespace(anchor_pos=object()),
    )


class NominalPhaseZeroContractTest(unittest.TestCase):
    def test_disables_randomness_but_preserves_terminations(self) -> None:
        cfg = _fake_env_cfg()
        termination_cfg = cfg.terminations
        event_cfg = cfg.events
        curriculum_cfg = cfg.curriculum

        audit = apply_nominal_phase_zero_contract(
            cfg,
            motion_file="/tmp/exact-long.npz",
            num_envs=1,
        )

        self.assertEqual(cfg.scene.num_envs, 1)
        self.assertEqual(cfg.commands.motion.motion_file, "/tmp/exact-long.npz")
        self.assertEqual(cfg.commands.motion.min_sample_idx, 0)
        self.assertEqual(cfg.commands.motion.max_sample_idx, 0)
        self.assertTrue(
            all(
                bounds == (0.0, 0.0)
                for bounds in cfg.commands.motion.pose_range.values()
            )
        )
        self.assertTrue(
            all(
                bounds == (0.0, 0.0)
                for bounds in cfg.commands.motion.velocity_range.values()
            )
        )
        self.assertEqual(cfg.commands.motion.joint_position_range, (0.0, 0.0))
        self.assertFalse(cfg.commands.motion.debug_vis)
        self.assertFalse(cfg.observations.policy.enable_corruption)
        self.assertIs(cfg.events, event_cfg)
        self.assertIs(cfg.curriculum, curriculum_cfg)
        self.assertTrue(all(value is None for value in vars(cfg.events).values()))
        self.assertTrue(all(value is None for value in vars(cfg.curriculum).values()))
        self.assertIs(cfg.terminations, termination_cfg)
        self.assertEqual(audit["start_phase"], 0)
        self.assertEqual(
            audit["disabled_event_terms"], ["physics_material", "push_robot"]
        )
        self.assertEqual(audit["disabled_curriculum_terms"], ["adr"])
        self.assertTrue(audit["hard_terminations_preserved"])


class InferenceResetContractTest(unittest.TestCase):
    def test_reset_and_reference_refresh_run_inside_inference_mode(self) -> None:
        calls: list[tuple[str, object]] = []

        class FakeEnv:
            def seed(self, seed: int) -> None:
                calls.append(("seed", (seed, torch.is_inference_mode_enabled())))

            def reset(self) -> tuple[torch.Tensor, dict[str, object]]:
                calls.append(("reset", torch.is_inference_mode_enabled()))
                return torch.zeros(1, 3), {}

        motion_command = object()

        def refresh(value: object) -> None:
            calls.append(
                (
                    "refresh",
                    (value is motion_command, torch.is_inference_mode_enabled()),
                )
            )

        obs = reset_episode_in_inference_mode(
            FakeEnv(),
            motion_command,
            seed=17,
            refresh_reference=refresh,
        )

        self.assertEqual(obs.shape, (1, 3))
        self.assertEqual(
            calls,
            [
                ("seed", (17, True)),
                ("reset", True),
                ("refresh", (True, True)),
            ],
        )


class EpisodeClassificationTest(unittest.TestCase):
    def test_all_exact_timeouts_complete(self) -> None:
        episodes = [
            {
                "steps": 272,
                "final_reference_phase": 271,
                "terminated": False,
                "timed_out": True,
                "all_numeric_finite": True,
            }
            for _ in range(3)
        ]

        result = classify_episodes(episodes, reference_states=272)

        self.assertEqual(result["outcome"], "source-completes-exact-long")
        self.assertTrue(result["all_episodes_complete"])
        self.assertEqual(result["survival_steps"], [272, 272, 272])
        self.assertEqual(result["expected_transition_count"], 271)
        self.assertEqual(result["expected_source_policy_steps"], 272)

    def test_all_early_terminations_fail(self) -> None:
        episodes = [
            {
                "steps": steps,
                "final_reference_phase": steps - 1,
                "terminated": True,
                "timed_out": False,
                "all_numeric_finite": True,
            }
            for steps in (103, 34, 85)
        ]

        result = classify_episodes(episodes, reference_states=272)

        self.assertEqual(result["outcome"], "source-fails-exact-long")
        self.assertFalse(result["any_episode_complete"])

    def test_mixed_completion_is_preserved(self) -> None:
        episodes = [
            {
                "steps": 272,
                "final_reference_phase": 271,
                "terminated": False,
                "timed_out": True,
                "all_numeric_finite": True,
            },
            {
                "steps": 90,
                "final_reference_phase": 89,
                "terminated": True,
                "timed_out": False,
                "all_numeric_finite": True,
            },
        ]

        result = classify_episodes(episodes, reference_states=272)

        self.assertEqual(result["outcome"], "mixed-source-competence")

    def test_nonfinite_or_inconsistent_episode_is_invalid(self) -> None:
        episodes = [
            {
                "steps": 272,
                "final_reference_phase": 270,
                "terminated": False,
                "timed_out": True,
                "all_numeric_finite": False,
            }
        ]

        result = classify_episodes(episodes, reference_states=272)

        self.assertEqual(result["outcome"], "invalid-execution")
        self.assertFalse(result["contract_valid"])

    def test_short_control_uses_explicit_outcome_labels(self) -> None:
        episodes = [
            {
                "steps": 125,
                "final_reference_phase": 124,
                "terminated": False,
                "timed_out": True,
                "all_numeric_finite": True,
            }
            for _ in range(3)
        ]

        result = classify_episodes(
            episodes,
            reference_states=125,
            outcome_label_set="short-control",
        )

        self.assertEqual(result["outcome"], "short-source-completes")
        self.assertEqual(result["outcome_label_set"], "short-control")
        self.assertEqual(result["expected_source_policy_steps"], 125)

    def test_unknown_outcome_label_set_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown outcome label set"):
            classify_episodes(
                [],
                reference_states=125,
                outcome_label_set="typo",
            )


if __name__ == "__main__":
    unittest.main()
