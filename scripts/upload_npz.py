import wandb
import os

REGISTRY_NAME = "motions"
COLLECTION_NAME = "waltz_full"
FILE_PATH = "/move/u/justingu/rmr_tracking/artifacts/waltz_full:v0/motion.npz"

# Log to a regular project, then link to registry
run = wandb.init(project="upload_npz", name=COLLECTION_NAME)

# Create a fresh artifact
artifact = wandb.Artifact(name=COLLECTION_NAME, type=REGISTRY_NAME)

# Add the file to the artifact, RENAME to "motion.npz"
if os.path.exists(FILE_PATH):
    print(f"Found file at {FILE_PATH}, adding as motion.npz")
    artifact.add_file(FILE_PATH, name="motion.npz")
else:
    raise FileNotFoundError(f"Could not find file at {FILE_PATH}")

logged_artifact = run.log_artifact(artifact)

# Link to the motions registry so training can find it
run.link_artifact(artifact=logged_artifact, target_path=f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}")
print(f"Artifact linked to registry: wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}")

run.finish()

