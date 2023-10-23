from argparse import Namespace
from copy import deepcopy
from typing import Optional

import wandb
from elk.evaluation.evaluate import Eval
from elk.training.sweep import Sweep
from elk.training.train import Elicit


def wandb_init_helper(
    args: Namespace, project_name: str = "elk_test_experiment"
) -> None:
    """
    Serializes args so they can be logged in wandb
    and starts a run according to the command type.
    """
    if args.wandb_tracking:
        if isinstance(args, Eval):
            args_serialized = deepcopy(args)
            args_serialized.out_dir = str(
                args_serialized.out_dir
            )  # .as_posix method would break on windows
            args_serialized.source = str(args_serialized.source)
            wandb.init(project=project_name, config=args_serialized, job_type="eval")
        elif isinstance(args, Elicit):
            wandb.init(project=project_name, config=args, job_type="train")
        elif isinstance(args, Sweep):
            wandb.init(
                project=project_name, config=args, job_type="sweep", group="Sweep1"
            )
    else:
        wandb.init(mode="disabled")


def find_run_id_by_name(entity_name: str, run_name: str) -> Optional[str]:
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
                return run.id

    # If no run found with the given name, return None
    return None
