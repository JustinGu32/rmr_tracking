"""Experimental runner correcting only the first observation of resumed learn()."""

from __future__ import annotations

from typing import Any

from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner

from .normalization import normalize_initial_observations


class InitialObservationNormalizedMotionOnPolicyRunner(MotionOnPolicyRunner):
    """Feed normalized actor/critic observations into the first resumed action."""

    def learn(
        self, num_learning_iterations: int, init_at_random_ep_len: bool = False
    ) -> Any:
        if self.current_learning_iteration <= 0:
            raise RuntimeError(
                "the corrected-order runner is restricted to a loaded resume checkpoint"
            )
        original_get_observations = self.env.get_observations
        call_count = 0

        def normalized_first_get_observations():
            nonlocal call_count
            observation, extras = original_get_observations()
            call_count += 1
            if call_count != 1:
                return observation, extras
            observation, extras, metadata = normalize_initial_observations(
                observation,
                extras,
                actor_normalizer=self.obs_normalizer,
                privileged_normalizer=self.privileged_obs_normalizer,
                privileged_observation_type=self.privileged_obs_type,
                device=self.device,
            )
            actor_count = getattr(self.obs_normalizer, "count", None)
            privileged_count = getattr(self.privileged_obs_normalizer, "count", None)
            print(
                "[RESUME-INITIAL-NORMALIZATION] "
                f"metadata={metadata} actor_count={actor_count} "
                f"privileged_count={privileged_count}",
                flush=True,
            )
            return observation, extras

        self.env.get_observations = normalized_first_get_observations
        try:
            result = super().learn(
                num_learning_iterations=num_learning_iterations,
                init_at_random_ep_len=init_at_random_ep_len,
            )
        finally:
            self.env.get_observations = original_get_observations
        if call_count < 1:
            raise RuntimeError("base learn() did not request an initial observation")
        return result
