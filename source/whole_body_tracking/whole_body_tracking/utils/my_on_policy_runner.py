import os

from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from isaaclab_rl.rsl_rl import export_policy_as_onnx

import wandb
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx
from whole_body_tracking.utils.holosoma_exporter import (
    attach_holosoma_metadata,
    export_holosoma_policy_as_onnx,
)


class MyOnPolicyRunner(OnPolicyRunner):
    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if self.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_policy_as_onnx(self.alg.policy, normalizer=self.alg.policy.actor_obs_normalizer, path=policy_path, filename=filename)
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device="cpu",
        registry_name: str = None,
        export_holosoma: bool | None = None,
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name
        # Enable holosoma export via constructor arg or EXPORT_HOLOSOMA env var
        if export_holosoma is not None:
            self.export_holosoma = export_holosoma
        else:
            self.export_holosoma = os.environ.get("EXPORT_HOLOSOMA", "0") == "1"

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if self.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_motion_policy_as_onnx(
                self.env.unwrapped, self.alg.policy, normalizer=self.alg.policy.actor_obs_normalizer, path=policy_path, filename=filename
            )
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))

            # --- Optional holosoma-native export ---
            if self.export_holosoma:
                holosoma_filename = filename.replace(".onnx", "_holosoma.onnx")
                try:
                    export_holosoma_policy_as_onnx(
                        self.env.unwrapped,
                        self.alg.policy,
                        normalizer=self.alg.policy.actor_obs_normalizer,
                        path=policy_path,
                        filename=holosoma_filename,
                    )
                    attach_holosoma_metadata(
                        self.env.unwrapped,
                        wandb.run.name,
                        path=policy_path,
                        filename=holosoma_filename,
                    )
                    wandb.save(
                        policy_path + holosoma_filename,
                        base_path=os.path.dirname(policy_path),
                    )
                except Exception as e:
                    print(f"[holosoma_exporter] WARNING: Holosoma export failed: {e}")

            # link the artifact registry to this run
            if self.registry_name is not None:
                wandb.run.use_artifact(self.registry_name)
                self.registry_name = None
