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
        # dataset_dir: str = field(
    #     default="data",
    #     metadata={"help": "Path to the folder containing the datasets."},
    # )
    # cutoff_len: int = field(
    #     default=1024,
    #     metadata={"help": "The cutoff length of the tokenized inputs in the dataset."},
    # )
    # train_on_prompt: bool = field(
    #     default=False,
    #     metadata={"help": "Whether or not to disable the mask on the prompt."},
    # )
    # mask_history: bool = field(
    #     default=False,
    #     metadata={"help": "Whether or not to mask the history and train on the last turn only."},
    # )
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
    # overwrite_cache: bool = field(
    #     default=False,
    #     metadata={"help": "Overwrite the cached training and evaluation sets."},
    # )
    # preprocessing_batch_size: int = field(
    #     default=1000,
    #     metadata={"help": "The number of examples in one group in pre-processing."},
    # )
    # preprocessing_num_workers: Optional[int] = field(
    #     default=None,
    #     metadata={"help": "The number of processes to use for the pre-processing."},
    # )
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes, truncate the number of examples for each dataset."},
    )
    # eval_num_beams: Optional[int] = field(
    #     default=None,
    #     metadata={"help": "Number of beams to use for evaluation. This argument will be passed to `model.generate`"},
    # )
    # ignore_pad_token_for_loss: bool = field(
    #     default=True,
    #     metadata={"help": "Whether or not to ignore the tokens corresponding to the pad label in loss computation."},
    # )
    test_size: float = field(
        default=0.0,
        metadata={"help": "Size of the development set, should be an integer or a float in range `[0,1)`."},
    )
    # packing: Optional[bool] = field(
    #     default=None,
    #     metadata={"help": "Enable sequences packing in training. Will automatically enable in pre-training."},
    # )
    # neat_packing: bool = field(
    #     default=False,
    #     metadata={"help": "Enable sequence packing without cross-attention."},
    # )
    # tool_format: Optional[str] = field(
    #     default=None,
    #     metadata={"help": "Tool format to use for constructing function calling examples."},
    # )
    # tokenized_path: Optional[str] = field(
    #     default=None,
    #     metadata={"help": "Path to save or load the tokenized datasets."},
    # )

    # def __init__(self):
    #     self.dataset_dir = None

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

        # if self.interleave_probs is not None:
        #     if self.mix_strategy == "concat":
        #         raise ValueError("`interleave_probs` is only valid for interleaved mixing.")
        #
        #     self.interleave_probs = list(map(float, split_arg(self.interleave_probs)))
        #     if self.dataset is not None and len(self.dataset) != len(self.interleave_probs):
        #         raise ValueError("The length of dataset and interleave probs should be identical.")
        #
        #     if self.eval_dataset is not None and len(self.eval_dataset) != len(self.interleave_probs):
        #         raise ValueError("The length of eval dataset and interleave probs should be identical.")

        if self.streaming and 1e-6 < self.test_size < 1:
            raise ValueError("Streaming mode should have an integer test size.")

        if self.streaming and self.max_samples is not None:
            raise ValueError("`max_samples` is incompatible with `streaming`.")

        # if self.mask_history and self.train_on_prompt:
        #     raise ValueError("`mask_history` is incompatible with `train_on_prompt`.")

