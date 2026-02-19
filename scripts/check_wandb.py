import wandb
import sys

def check_artifact(entity, project, collection, version="latest"):
    api = wandb.Api()
    path = f"{entity}/{project}/{collection}:{version}"
    print(f"Checking {path}...")
    try:
        artifact = api.artifact(path)
        print(f"FAILED to error (wait, success?): Found artifact {artifact.name}")
        return True
    except Exception as e:
        print(f"Error checking {path}: {e}")
        return False

if __name__ == "__main__":
    # Test the one the user used
    # robot-mcrobotface/wandb-registry-motions/tracking_chair_step.npz:v0
    print("--- Check 1: User provided path ---")
    check_artifact("robot-mcrobotface", "wandb-registry-motions", "tracking_chair_step.npz", "v0")

    # Test the one from previous step
    # robot-mcrobotface/wandb-registry-motions/chair_step_truncated_converted.npz:v0
    print("\n--- Check 2: Likely correct path ---")
    check_artifact("robot-mcrobotface", "wandb-registry-motions", "chair_step_truncated_converted.npz", "latest")

    # Test with user entity
    print("\n--- Check 3: User entity (kkarenvoo) ---")
    check_artifact("kkarenvoo", "wandb-registry-motions", "chair_step_truncated_converted.npz", "latest")
    
    # Test source project
    print("\n--- Check 4: Source project (csv_to_npz) ---")
    check_artifact("robot-mcrobotface", "csv_to_npz", "chair_step_truncated_converted.npz", "latest")
    check_artifact("kkarenvoo", "csv_to_npz", "chair_step_truncated_converted.npz", "latest")
    
    # List collections in the project
    print("\n--- Listing Collections ---")
    try:
        api = wandb.Api()
        collections = api.artifact_type("motion", "robot-mcrobotface/wandb-registry-motions").collections()
        for c in collections:
            print(f"Found collection: {c.name}")
    except Exception as e:
        print(f"Could not list collections: {e}")
