"""Functions for extracting the hidden states of a model."""
import logging
import os
from contextlib import nullcontext, redirect_stdout
from dataclasses import InitVar, dataclass, replace
from itertools import zip_longest
from typing import Any, Iterable, Literal
from warnings import filterwarnings

from transformers import AutoTokenizer, AutoModelForCausalLM

import torch
from datasets import (
    Array2D,
    Array3D,
    DatasetDict,
    DatasetInfo,
    DownloadMode,
    Features,
    Sequence,
    SplitDict,
    SplitInfo,
    Value,
    get_dataset_config_info,
)
from simple_parsing import Serializable, field
from torch import Tensor
from transformers import AutoConfig, PreTrainedModel

from ..promptsource import DatasetTemplates
from ..utils import (
    Color,
    assert_type,
    colorize,
    float_to_int16,
    infer_label_column,
    infer_num_classes,
    instantiate_model,
    instantiate_tokenizer,
    is_autoregressive,
    prevent_name_conflicts,
    select_split,
    select_train_val_splits,
    select_usable_devices,
)
from .dataset_name import (
    DatasetDictWithName,
    parse_dataset_string,
)
from .generator import _GeneratorBuilder
from .prompt_loading import load_prompts

@dataclass
class Extract(Serializable):
    """Config for extracting hidden states from a language model."""

    model: str = field(positional=True)
    """HF model string identifying the language model to extract hidden states from."""

    datasets: tuple[str, ...] = field(positional=True)
    """Names of HF datasets to use, e.g. `"super_glue:boolq"` or `"imdb"`"""

    data_dirs: tuple[str, ...] = ()
    """Directory to use for caching the hiddens. Defaults to `HF_DATASETS_CACHE`."""

    binarize: bool = False
    """Whether to binarize the dataset labels for multi-class datasets."""

    int8: bool = False
    """Whether to perform inference in mixed int8 precision with `bitsandbytes`."""

    max_examples: tuple[int, int] = (1000, 1000)
    """Maximum number of examples to use from each split of the dataset."""

    num_shots: int = 0
    """Number of examples for few-shot prompts. If zero, prompts are zero-shot."""

    num_variants: int = -1
    """The number of prompt templates to use for each example. If -1, all available
    templates are used."""

    layers: tuple[int, ...] = ()
    """Indices of layers to extract hidden states from. We ignore the embedding,
    have only the output of the transformer layers."""

    layer_stride: InitVar[int] = 1
    """Shortcut for `tuple(range(1, num_layers, stride))`."""

    seed: int = 42
    """Seed to use for prompt randomization. Defaults to 42."""

    template_path: str | None = None
    """Path to pass into `DatasetTemplates`. By default we use the dataset name."""

    token_loc: Literal["first", "last", "mean"] = "last"
    """The location of the token to extract hidden states from."""

    use_encoder_states: bool = False
    """Whether to extract hidden states from the encoder instead of the decoder in the
    case of encoder-decoder models."""

    def __post_init__(self, layer_stride: int):
        if self.num_variants != -1:
            print("WARNING: num_variants is deprecated; use prompt_indices instead.")
        if len(self.datasets) == 0:
            raise ValueError(
                "Must specify at least one dataset to extract hiddens from."
            )

        if len(self.max_examples) > 2:
            raise ValueError(
                "max_examples should be a list of length 0, 1, or 2,"
                f"but got {len(self.max_examples)}"
            )
        if not self.max_examples:
            self.max_examples = (int(1e100), int(1e100))

        # Broadcast the dataset name to all data_dirs
        if len(self.data_dirs) == 1:
            self.data_dirs *= len(self.datasets)
        elif self.data_dirs and len(self.data_dirs) != len(self.datasets):
            raise ValueError(
                "data_dirs should be a list of length 0, 1, or len(datasets),"
                f" but got {len(self.data_dirs)}"
            )

        if self.layers and layer_stride > 1:
            raise ValueError(
                "Cannot use both --layers and --layer-stride. Please use only one."
            )
        elif layer_stride > 1:
            from transformers import AutoConfig, PretrainedConfig

            # Look up the model config to get the number of layers
            config = assert_type(
                PretrainedConfig, AutoConfig.from_pretrained(self.model)
            )
            layer_range = range(1, config.num_hidden_layers, layer_stride)
            self.layers = tuple(layer_range)

    def explode(self) -> list["Extract"]:
        """Explode this config into a list of configs, one for each layer."""
        return [
            replace(self, datasets=(ds,), data_dirs=(data_dir,) if data_dir else ())
            for ds, data_dir in zip_longest(self.datasets, self.data_dirs)
        ]
    
class InitialEmbedding(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.word_embed = model.transformer.wte
        self.pos_embed = model.transformer.wpe
    
    def forward(self, input_ids):
        # TODO : check that
        pos_ids = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        return self.word_embed(input_ids) + self.pos_embed(pos_ids)

def get_embed(model):
    return InitialEmbedding(model)

def get_block(model, layer):
    return model.transformer.h[layer]


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
def get_acts(statements, tokenizer, model, batch_size=32, layers=None, intermediate_device="cpu", compute_device=DEVICE):
    """
    Get given layer activations for the statements.
    Return dictionary of stacked activations.

    Caution: Layer 0 is embedding layer, layer 1 is the first transformer layer, so model.transformer.h[0]
    """
    model.eval().to(intermediate_device)
    if layers is None:
        layers = list(range(model.config.num_hidden_layers + 1))

    # get last token indexes for all statements
    last_tokens = [len(tokenizer.encode(statement)) - 1 for statement in statements]

    #print(last_tokens)

    current_hiddens = []
    all_hiddens = [[] for _ in range(model.config.num_hidden_layers + 1)]

    embed = get_embed(model).to(compute_device)

    bos_token = tokenizer.bos_token_id
    
    for batch_start in range(0, len(statements), batch_size):
        batch = statements[batch_start:min(batch_start + batch_size, len(statements))]
        # TODO : check for last token (should be ".")
        input_ids = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).input_ids.to(compute_device)

        if bos_token is not None and bos_token != input_ids[0, 0]:
            input_ids = torch.cat([torch.zeros(input_ids.size(0), 1, device=input_ids.device, dtype=input_ids.dtype).fill_(bos_token), input_ids], dim=1)

        current_hiddens.append(embed(input_ids))
        if 0 in layers:
            all_hiddens[0].append(current_hiddens[-1][torch.arange(input_ids.size(0)), last_tokens[batch_start:batch_start + input_ids.size(0)]].to(intermediate_device))
        
    embed.to(intermediate_device)

    for block_idx in range(max(layers)):
        block = get_block(model, block_idx).to(compute_device)

        for batch_idx, batch in enumerate(current_hiddens):
            out = block(batch)[0]
            if block_idx + 1 in layers:
                all_hiddens[block_idx + 1].append(
                    out[
                        torch.arange(input_ids.size(0)),
                        last_tokens[batch_idx * batch_size:min((batch_idx + 1) * batch_size, len(statements))]
                    ].to(intermediate_device)
                )
            current_hiddens[batch_idx] = out

        block.to(intermediate_device)
    
    return {layer: torch.cat(acts) for layer, acts in enumerate(all_hiddens) if len(acts) > 0}


@torch.no_grad()
def generate_acts(
    cfg,
    layers=None,
    split_type = "train",
    rank=0,
    world_size=1,
    noperiod=False,
    intermediate_device="cpu",
    compute_device=DEVICE,
):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    model = AutoModelForCausalLM.from_pretrained(cfg.model).to(intermediate_device)
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token = tokenizer.eos_token
    
    if layers is None:
        layers = list(range(model.config.num_hidden_layers + 1))

    ds_names = cfg.datasets

    prompt_ds = load_prompts(
        ds_names[0],
        binarize=cfg.binarize,
        num_shots=cfg.num_shots,
        split_type=split_type,
        template_path=cfg.template_path,
        rank=rank,
        world_size=world_size,
        seed=cfg.seed,
    )
    
    num_yielded = 0
    for example_id, example in enumerate(prompt_ds):
        num_variants = len(example["prompts"])
        num_choices = len(example["prompts"][0])

        hidden_dict = {
            f"hidden_{layer_idx}": torch.empty(
                num_variants,
                num_choices,
                model.config.hidden_size,
                device=intermediate_device,
                dtype=torch.int16,
            )
            for layer_idx in layers
        }
        
        text_questions = []
        statements = []
        for i, record in enumerate(example["prompts"]):
            variant_questions = []

            # Iterate over answers
            for j, choice in enumerate(record):
                text = choice["question"]

                variant_questions.append(
                    dict(
                        {
                            "template_id": i,
                            "template_name": example["template_names"][i],
                            "text": dict(
                                {
                                    "question": text,
                                    "answer": choice["answer"],
                                }
                            ),
                            "example_id": example_id,
                        }
                    )
                )
                statements.append(choice["question"] + " " + choice["answer"])
            
            text_questions.append(variant_questions)
        
        acts = get_acts(statements, tokenizer, model, layers=layers, intermediate_device=intermediate_device, compute_device=compute_device)

        # Fill hidden_dict with activations
        for layer_idx, act in acts.items():
            idx = 0
            for i, record in enumerate(example["prompts"]):
                for j, choice in enumerate(record):
                    hidden_dict[f"hidden_{layer_idx}"][i, j] = act[idx]
                    idx += 1
            
        # We skipped a variant because it was too long; move on to the next example
        if len(text_questions) != num_variants:
            continue

        out_record = dict()
        out_record["label"] = example["label"]
        out_record["variant_ids"] = example["template_names"]
        out_record["text_questions"] = text_questions
        for layer_idx, hidden in hidden_dict.items():
            out_record[layer_idx] = hidden

        num_yielded += 1
        yield out_record


@torch.inference_mode()
def extract_hiddens(
    cfg: "Extract",
    *,
    device: str | torch.device = "cpu",
    split_type: Literal["train", "val"] = "train",
    rank: int = 0,
    world_size: int = 1,
) -> Iterable[dict]:
    """Run inference on a model with a set of prompts, yielding the hidden states."""
    yield from generate_acts(cfg, split_type=split_type, rank=rank, world_size=world_size)


# Dataset.from_generator wraps all the arguments in lists, so we unpack them here
def _extraction_worker(**kwargs):
    #cfg = kwargs["cfg"][0]
    yield from extract_hiddens(**{k: v[0] for k, v in kwargs.items()})


def hidden_features(cfg: Extract) -> tuple[DatasetInfo, Features]:
    """Return the HuggingFace `Features` corresponding to an `Extract` config."""
    with prevent_name_conflicts():
        model_cfg = AutoConfig.from_pretrained(cfg.model)

    ds_name, config_name = parse_dataset_string(dataset_config_str=cfg.datasets[0])
    info = get_dataset_config_info(ds_name, config_name or None)

    if not cfg.template_path:
        prompter = DatasetTemplates(ds_name, config_name)
    else:
        prompter = DatasetTemplates(cfg.template_path)

    ds_features = assert_type(Features, info.features)
    label_col = prompter.label_column or infer_label_column(ds_features)
    num_classes = (
        2
        if cfg.binarize or prompter.binarize
        else infer_num_classes(ds_features[label_col])
    )

    num_dropped = prompter.drop_non_mc_templates()
    num_variants = len(prompter.templates)
    if num_dropped:
        print(f"Dropping {num_dropped} non-multiple choice templates")

    layer_indices = cfg.layers or tuple(range(1, model_cfg.num_hidden_layers))
    layer_cols = {
        f"hidden_{layer}": Array3D(
            dtype="int16",
            shape=(num_variants, num_classes, model_cfg.hidden_size),
        )
        for layer in layer_indices
    }
    other_cols = {
        "variant_ids": Sequence(
            Value(dtype="string"),
            length=num_variants,
        ),
        "label": Value(dtype="int64"),
        "text_questions": Sequence(
            Sequence(
                Value(dtype="string"),
            ),
            length=num_variants,
        ),
    }

    # Only add model_logits if the model is an autoregressive model
    if is_autoregressive(model_cfg, not cfg.use_encoder_states):
        other_cols["model_logits"] = Array2D(
            shape=(num_variants, num_classes),
            dtype="float32",
        )

    return info, Features({**layer_cols, **other_cols})


def extract(
    cfg: "Extract",
    *,
    disable_cache: bool = False,
    highlight_color: Color = "cyan",
    num_gpus: int = -1,
    min_gpu_mem: int | None = None,
    split_type: Literal["train", "val", None] = None,
) -> DatasetDictWithName:
    """Extract hidden states from a model and return a `DatasetDict` containing them."""
    info, features = hidden_features(cfg)

    devices = select_usable_devices(num_gpus, min_memory=min_gpu_mem)
    limits = cfg.max_examples
    splits = assert_type(SplitDict, info.splits)

    pretty_name = colorize(assert_type(str, cfg.datasets[0]), highlight_color)
    if split_type is None:
        train, val = select_train_val_splits(splits)

        print(f"{pretty_name} using '{train}' for training and '{val}' for validation")
        splits = SplitDict({train: splits[train], val: splits[val]})
        split_types = ["train", "val"]
    else:
        # Remove the split we're not using
        limits = [limits[0]] if split_type == "train" else limits
        split_name = select_split(splits, split_type)
        splits = SplitDict({split_name: splits[split_name]})
        split_types = [split_type]

        if split_type == "train":
            print(f"{pretty_name} using '{split_name}' for training")
        else:
            print(f"{pretty_name} using '{split_name}' for validation")

    builders = {
        split_name: _GeneratorBuilder(
            cache_dir=None,
            features=features,
            generator=_extraction_worker,
            split_name=split_name,
            split_info=SplitInfo(
                name=split_name,
                num_examples=min(limit, v.num_examples) * len(cfg.datasets),
                dataset_name=v.dataset_name,
            ),
            gen_kwargs=dict(
                cfg=[cfg] * len(devices),
                device=devices,
                rank=list(range(len(devices))),
                split_type=[ty] * len(devices),
                world_size=[len(devices)] * len(devices),
            ),
        )
        for limit, (split_name, v), ty in zip(limits, splits.items(), split_types)
    }
    import multiprocess as mp

    mp.set_start_method("spawn", force=True)  # type: ignore[attr-defined]

    ds = dict()
    for split, builder in builders.items():
        builder.download_and_prepare(
            download_mode=DownloadMode.FORCE_REDOWNLOAD if disable_cache else None,
            num_proc=len(devices),
        )
        ds[split] = builder.as_dataset(split=split)

    dataset_dict = DatasetDict(ds)
    return DatasetDictWithName(
        name=cfg.datasets[0],
        dataset=dataset_dict,
    )
