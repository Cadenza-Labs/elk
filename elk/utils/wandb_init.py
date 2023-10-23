from argparse import Namespace
from copy import deepcopy

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
