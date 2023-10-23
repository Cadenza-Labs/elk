from pathlib import Path
from typing import Optional, Tuple

import wandb


def find_run_details(entity_name: str, run_name: str) -> Optional[Tuple[str, str]]:
    """
    Search through all projects of a given entity to find the run ID
    corresponding to a given run name.
    This requires run names to be always unique.

    Args:
    - entity_name (str): Name of the entity (user/team).
    - run_name (str): Name of the run.

    Returns:
    - str: run_id if found, otherwise None.
    """

    # Set up wandb API
    api = wandb.Api()

    # Iterate over all projects of the entity
    for project in api.projects(entity=entity_name):
        project_name = project.name

        # Search for the run by name in the current project
        runs = api.runs(path=f"{entity_name}/{project_name}")

        for run in runs:
            if run.name == run_name:
                return project_name, run.id

    # If no run found with the given name, return None
    return None


def wandb_save_probes(out_dir: Path) -> None:
    if wandb.run is not None:
        for dir in ["lr_models", "reporters"]:
            artifact = wandb.Artifact(dir, type="model")
            artifact.add_dir(out_dir / dir)
            wandb.run.log_artifact(artifact)
