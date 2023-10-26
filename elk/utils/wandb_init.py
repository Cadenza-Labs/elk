import os
from argparse import Namespace

import wandb
from elk.evaluation.evaluate import Eval
from elk.training.sweep import Sweep
from elk.training.train import Elicit


def wandb_init_helper(args: Namespace) -> None:
    """
    Serializes args so they can be logged in wandb
    and starts a run according to the command type.
    """
    project_name = args.wandb_project_name
    if args.wandb_tracking:
        # Get entity name
        entity_name = os.getenv("WANDB_ENTITY")
        assert entity_name is not None, "Please set a WANDB_ENTITY env variable"

        if isinstance(args, Eval):
            args_serialized = args.to_dict()
            args_serialized.out_dir = str(
                args_serialized.out_dir
            )  # .as_posix method would break on windows
            args_serialized.source = str(args_serialized.source)
            wandb.init(
                entity=entity_name,
                project=project_name,
                config=args_serialized,
                job_type="eval",
            )
        elif isinstance(args, Elicit):
            wandb.init(
                entity=entity_name, project=project_name, config=args, job_type="train"
            )
        elif isinstance(args, Sweep):
            wandb.init(
                entity=entity_name, project=project_name, config=args, job_type="sweep"
            )
    else:
        wandb.init(mode="disabled")
