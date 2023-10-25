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
        if project_name is None:
            project_name = "default_project"

        if isinstance(args, Eval):
            args_serialized = args.to_dict()
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
