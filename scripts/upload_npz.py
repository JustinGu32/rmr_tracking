import wandb

REGISTRY_NAME = "motions"
COLLECTION_NAME = "takara_walk_isaac"

run = wandb.init(project="takara_walk_isaac", name=COLLECTION_NAME)

logged_artifact = run.log_artifact(artifact_or_path="./motions/takara_walk_isaac/motion.npz", name=COLLECTION_NAME, type=REGISTRY_NAME)

run.link_artifact(artifact=logged_artifact, target_path=f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}")
