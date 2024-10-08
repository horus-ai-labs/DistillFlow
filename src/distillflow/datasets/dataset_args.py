from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Optional, Literal, Dict, Any, List, Union

from datasets import Dataset

from .template import Template

@dataclass
class DatasetArgs:
    path: str = ""
    dataset: Dataset = None
    num_samples: Optional[int] = None
    load_from_cache_file: bool = True
    template: Optional[Template] = None
    split: str = "train"

@dataclass
class DataArgs:
    r"""
    Arguments pertaining to what data we are going to input our model for training and evaluation.
    """
    seed: Optional[int] = field(
        default=0,
        metadata={"help": "Seed to shuffle the dataset."},
    )
    train_datasets: Optional[List[DatasetArgs]] = field(
        default=None,
        metadata={"help": "The dataset(s) to use for training. Provide as a list of DatasetArgs."},
    )
    eval_datasets: Optional[List[DatasetArgs]] = field(
        default=None,
        metadata={"help": "The names of dataset(s) to use for evaluation. Provide as a list of DatasetArgs."},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pre-trained models downloaded from huggingface.co or modelscope.cn."},
    )
    hf_hub_token: Optional[str] = field(
        default=None,
        metadata={"help": "Auth token to log in with Hugging Face Hub."},
    )
    text_field: Optional[str] = field(
        default="text",
        metadata={"help": "Name of the field key to convert the dataset."},
    )
    streaming: bool = field(
        default=False,
        metadata={"help": "Enable dataset streaming."},
    )
    buffer_size: Optional[int] = field(
        default=16384,
        metadata={"help": "Size of the buffer to randomly sample examples from in dataset streaming."},
    )
    mix_strategy: Optional[Literal["concat", "interleave_under", "interleave_over"]] = field(
        default="concat",
        metadata={"help": "Strategy to use in dataset mixing (concat/interleave) (undersampling/oversampling)."},
    )
    interleave_probs: Optional[str] = field(
        default=None,
        metadata={"help": "Probabilities to sample data from datasets. Use commas to separate multiple datasets."},
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes, truncate the number of examples for each dataset."},
    )
    test_size: float = field(
        default=0.0,
        metadata={"help": "Size of the development set, should be an integer or a float in range `[0,1)`."},
    )

    def __post_init__(self):
        def split_arg(arg):
            if isinstance(arg, str):
                return [item.strip() for item in arg.split(",")]
            return arg

        self.train_datasets = split_arg(self.train_datasets)
        self.eval_datasets = split_arg(self.eval_datasets)

        if self.train_datasets is None and self.test_size > 1e-6:
            raise ValueError("Cannot specify `val_size` if `dataset` is None.")

        if self.eval_datasets is not None and self.test_size > 1e-6:
            raise ValueError("Cannot specify `val_size` if `eval_dataset` is not None.")

        if self.streaming and 1e-6 < self.test_size < 1:
            raise ValueError("Streaming mode should have an integer test size.")

        if self.streaming and self.max_samples is not None:
            raise ValueError("`max_samples` is incompatible with `streaming`.")
