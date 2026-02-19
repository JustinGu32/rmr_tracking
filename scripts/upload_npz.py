import wandb
import os

REGISTRY_NAME = "motions"
COLLECTION_NAME = "chair_step"
FILE_PATH = "/move/u/karenvo/Projects/rmr_tracking/artifacts/chair_step_truncated_converted.npz"

# We remove entity argument to let it default to robot-mcrobotface (which works)
run = wandb.init(project="chair_step", name=COLLECTION_NAME)

# Create a fresh artifact
artifact = wandb.Artifact(name=COLLECTION_NAME, type=REGISTRY_NAME)

# Add the file to the artifact, RENAME to "motion.npz"
if os.path.exists(FILE_PATH):
    print(f"Found file at {FILE_PATH}, adding as motion.npz")
    artifact.add_file(FILE_PATH, name="motion.npz")
else:
    raise FileNotFoundError(f"Could not find file at {FILE_PATH}")

logged_artifact = run.log_artifact(artifact)

# We don't need to link if we just use the artifact from the project directly.
# The training script uses robot-mcrobotface/chair_step/chair_step:latest
print(f"Artifact logged to project chair_step with name {COLLECTION_NAME}")

run.finish()

