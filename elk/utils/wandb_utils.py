from pathlib import Path, PosixPath
from typing import List, Literal, Optional, Tuple

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


def wandb_save_probes_dir(out_dir: Path, model_dir: str) -> None:
    if wandb.run is not None:
        artifact_name = get_artifact_dir(out_dir / model_dir)
        print(f"Saving artifact with name {artifact_name} and type {model_dir}")
        artifact = wandb.Artifact(artifact_name, type=model_dir)
        artifact.add_dir(out_dir / model_dir)
        wandb.run.log_artifact(artifact)


def wandb_rename_run(out_dir: Path) -> Optional[str]:
    "Highly hacky way to rename a run."
    str_out_dir = str(out_dir)
    if wandb.run is not None:
        if "sweeps" in str_out_dir:
            return str_out_dir.split("sweeps/")[-1].split("/")[0]
        else:
            return str_out_dir.split("/")[-1]


def get_artifact_dir(path: Path, run_name: Optional[str] = None) -> str:
    "Get the wandb name of a model from its local path."

    # If path contains suffix, remove it
    path = path.with_suffix("")

    if run_name is None:
        if wandb.run is not None:
            run_name = wandb.run.name
        else:
            raise ValueError("Wandb run is not running and not run_name provided.")
    name = []
    while ((name_part := path.name) != run_name) and (name_part != ""):
        name = [name_part] + name
        path = path.parent
    name = [run_name] + name
    return ".".join(name)


def wandb_download_probe(
    entity_name: str,
    project_name: str,
    run_name: str,
    model_dir: Literal["lr_models", "reporters"],
    layers: List[int],
    artifact_path: Optional[str] = None,
    model_name: Optional[str] = None,
    dataset_name: Optional[str] = None,
    verbose: int = 0,
) -> PosixPath:
    """
    Downloads probes from specified project and run.
    Requires model_name and dataset_name for sweeps.
    """
    assert len(layers) > 0, "List of layers cannot be empty."

    # Find run_id of the run
    api = wandb.Api()
    _, run_id = find_run_details(entity_name, run_name)
    if run_id is None:
        raise ValueError(f"Run {run_name} not found.")
    run = api.run(f"{entity_name}/{project_name}/{run_id}")

    # Depending on the job type, change the probe_name
    if run.job_type == "train":
        probe_name = run_name + "." + model_dir + "." + "layer_{layer}.pt:latest"
    elif run.job_type == "sweep":
        assert model_name is not None, "model_name has to be specified for sweep runs"
        assert (
            dataset_name is not None
        ), "dataset_name has to be specified for sweep runs"
        probe_name = ".".join(
            [
                run_name,
                model_dir,
                dataset_name,
                model_name,
                "layer_{layer}.pt:latest",
            ]
        )
    else:
        raise ValueError("Job type not supported.")

    # Download the probes in layers
    for layer in layers:
        artifact_name = f"{entity_name}/{project_name}/{probe_name.format(layer=layer)}"
        artifact = api.artifact(artifact_name)
        if artifact_path is None:
            artifact_path = f"artifacts/{run_name}"
        artifact_dir = PosixPath(artifact.download(artifact_path))
        if verbose > 0:
            print(
                f"Probe {probe_name.format(layer=layer)} downloaded to {artifact_dir}."
            )
    return artifact_dir
