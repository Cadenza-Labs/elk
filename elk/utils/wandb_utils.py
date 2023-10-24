from pathlib import Path
from typing import Optional, Tuple

import wandb


def find_run_details(
    entity_name: str, run_name: str
) -> Tuple[Optional[str], Optional[str]]:
    """
    Search through all projects of a given entity to find the run ID
    corresponding to a given run name.
    This requires run names to be ALWAYS unique,
    which essentially requires all the experiments be run on the same machine.
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
    return None, None


def wandb_save_probes(out_dir: Path) -> None:
    if wandb.run is not None:
        for dir in ["lr_models", "reporters"]:
            artifact = wandb.Artifact(dir, type="model")
            artifact.add_dir(out_dir / dir)
            wandb.run.log_artifact(artifact)
