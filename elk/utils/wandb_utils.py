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


def wandb_save_probe(probe_path: Path, model_type: str) -> None:
    if wandb.run is not None:
        probe_name = (
            get_model_name(probe_path) + "." + probe_path.name
        )  # can break on windows
        artifact = wandb.Artifact(probe_name, type=model_type)
        artifact.add_file(probe_path)
        wandb.run.log_artifact(artifact)


def wandb_rename_run(out_dir: Path) -> Optional[str]:
    "Highly hacky way to rename a run."
    str_out_dir = str(out_dir)
    if wandb.run is not None:
        if "sweeps" in str_out_dir:
            return str_out_dir.split("sweeps/")[-1].split("/")[0]
        else:
            return str_out_dir.split("/")[-1]


def get_model_name(path: Path) -> str:
    "Get the wandb name of a model from its local path."
    if wandb.run is not None:
        run_name = wandb.run.name
        name = []
        while ((name_part := path.parent.name) != run_name) and (name_part != ""):
            name = name + [name_part]
            path = path.parent
        name = [run_name] + name
        return ".".join(name)
    raise ValueError("Wandb run is not running.")
