"""Extract hidden states from a model."""

from .extraction import extract_hiddens, PromptCollator
from ..files import args_to_uuid, elk_cache_dir
from ..training.preprocessing import silence_datasets_messages
from transformers import AutoConfig, AutoTokenizer
import json
from datasets import Dataset


def run(args):
    """Run the extraction subcommand for ELK.

    This function is called upon running `elk extract`.
    """

    def extract(args, split: str):
        """Extract hidden states for a given split.

        The split can be "train", "val", or "test".
        First the prompts are generated by the PromptCollator.
        Then, the hidden states are extracted by the extract_hiddens function.
        Finally, the hidden states and labels are saved to disk.
        """
        frac = 1 - args.val_frac if split == "train" else args.val_frac

        collator = PromptCollator(
            *args.dataset,
            max_examples=round(args.max_examples * frac) if args.max_examples else 0,
            split=split,
            label_column=args.label_column,
            num_shots=args.num_shots,
            strategy=args.prompts,
            balance=args.balance,
        )

        if split == "train":
            prompt_names = collator.prompter.all_template_names
            if args.prompts == "all":
                print(f"Using {len(prompt_names)} prompts per example: {prompt_names}")
            elif args.prompts == "randomize":
                print(f"Randomizing over {len(prompt_names)} prompts: {prompt_names}")
            else:
                raise ValueError(f"Unknown prompt strategy: {args.prompts}")

        return Dataset.from_generator(
            extract_hiddens,
            gen_kwargs={
                "model_str": args.model,
                "tokenizer": tokenizer,
                "collator": collator,
                "layers": args.layers,
                "prompt_suffix": args.prompt_suffix,
                "token_loc": args.token_loc,
                "use_encoder_states": args.use_encoder_states,
            },
        )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # If the user didn't specify a name, we'll use a hash of the CLI args
    if not args.name:
        args.name = args_to_uuid(args)

    save_dir = elk_cache_dir() / args.name
    print(f"Saving results to \033[1m{save_dir}\033[0m")  # bold

    print("Loading datasets")
    silence_datasets_messages()

    train_dset = extract(args, "train")
    valid_dset = extract(args, "validation")

    with open(save_dir / "args.json", "w") as f:
        json.dump(vars(args), f)

    with open(save_dir / "model_config.json", "w") as f:
        config = AutoConfig.from_pretrained(args.model)
        json.dump(config.to_dict(), f)

    return train_dset, valid_dset
