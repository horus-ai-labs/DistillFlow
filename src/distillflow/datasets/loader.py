import json
import os
from functools import partial
from typing import TypedDict, Optional, Dict, Any, Sequence, Literal, Union, List

import numpy as np
from datasets import DatasetDict, load_dataset, Dataset, IterableDataset, concatenate_datasets, interleave_datasets
from transformers import PreTrainedTokenizer, training_args
from transformers.utils import cached_file

from .dataset_args import DatasetArgs, DataArgs
from ..common import get_logger
from ..model.args import ModelArguments

DATA_CONFIG = "dataset_info.json"

logger = get_logger(__name__)

class DatasetModule(TypedDict):
    train_dataset: Optional[Dataset]
    eval_dataset: Optional[Dataset]


def _load_dataset(
        dataset_args: DatasetArgs,
        data_args: DataArgs,
        tokenizer: PreTrainedTokenizer) -> Dataset:

    if dataset_args is not None:
        logger.info("Loading dataset {}...".format(dataset_args))
        dataset = load_dataset(
            path=dataset_args.path,
            split=dataset_args.split,
            cache_dir=data_args.cache_dir,
            token=data_args.hf_hub_token,
            streaming=data_args.streaming, # and (dataset_attr.load_from != "file")),
            trust_remote_code=True
        )
        # Shuffle dataset with a pre-seed
        dataset = dataset.shuffle(seed=data_args.seed)

        # if data_args.streaming and (dataset_attr.load_from == "file"):  # faster than specifying streaming=True
        #     dataset = dataset.to_iterable_dataset()  # TODO: add num shards parameter

        if dataset_args.num_samples is not None:# and not data_args.streaming:
            target_num = dataset_args.num_samples
            indexes = np.random.permutation(len(dataset))[:target_num]  # all samples should be included
            target_num -= len(indexes)
            if target_num > 0:
                expand_indexes = np.random.choice(len(dataset), target_num)
                indexes = np.concatenate((indexes, expand_indexes), axis=0)

            assert len(indexes) == dataset_args.num_samples, "Sample num mismatched."
            dataset = dataset.select(indexes)
            logger.info("Sampled {} examples from dataset {}.".format(dataset_args.num_samples, dataset_args))

        if data_args.max_samples is not None:  # truncate dataset
            max_samples = min(data_args.max_samples, len(dataset))
            dataset = dataset.select(range(max_samples))

        column_names = list(next(iter(dataset)).keys())

        dataset = dataset.map(
            partial(dataset_args.template.convert),
            batched=True,
            remove_columns=column_names,
            load_from_cache_file=dataset_args.load_from_cache_file
        )

        if data_args.text_field is not None:
            dataset = dataset.map(partial(to_text, data_args.text_field, tokenizer), batched=True, load_from_cache_file=dataset_args.load_from_cache_file, remove_columns=dataset.column_names)
        return dataset

def to_text(field_name, tokenizer: PreTrainedTokenizer, example: Dict[str, Any]) -> Dict[str, Any]:
    system = example["_system"]
    prompt = example["_prompt"]
    response = example["_response"]
    message = system + prompt + response
    return {field_name: tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)}

def split_dataset(dataset: Dataset, data_args: DataArgs, seed: int) -> DatasetDict:
    r"""
    Splits the dataset and returns a dataset dict containing train set and validation set.

    Supports both map dataset and iterable dataset.
    """
    if data_args.streaming:
        dataset = dataset.shuffle(buffer_size=data_args.buffer_size, seed=seed)
        val_set = dataset.take(int(data_args.test_size))
        train_set = dataset.skip(int(data_args.test_size))
        return DatasetDict({"train": train_set, "validation": val_set})
    else:
        dataset = dataset.shuffle(seed=seed)
        test_size = int(data_args.test_size) if data_args.test_size > 1 else data_args.test_size
        dataset = dataset.train_test_split(test_size=test_size, seed=seed)
        return DatasetDict({"train": dataset["train"], "validation": dataset["test"]})

def get_dataset(data_args: DataArgs,
                tokenizer: PreTrainedTokenizer,
                tokenize=False,
                tokenizer_function=None) -> DatasetModule:
    # Load and preprocess dataset
    # with training_args.main_process_first(desc="load dataset"):
    dataset = _get_merged_dataset(data_args.train_datasets, data_args, tokenizer)
    dataset = dataset.shuffle(seed=data_args.seed)
    eval_dataset = None
    if data_args.eval_datasets:
        eval_dataset = _get_merged_dataset(data_args.eval_datasets, data_args, tokenizer)

    dataset_dict = {}

    if eval_dataset is None:
        dataset_dict = split_dataset(dataset, data_args, data_args.seed)
    else:
        if data_args.streaming:
            eval_dataset = eval_dataset.shuffle(buffer_size=data_args.buffer_size, seed=data_args.seed)

        dataset_dict["validation"] = eval_dataset

        if dataset is not None:
            if data_args.streaming:
                dataset = dataset.shuffle(buffer_size=data_args.buffer_size, seed=data_args.train_datasets.seed)

            dataset_dict["train"] = dataset

        dataset_dict = DatasetDict(dataset_dict)

    dataset_module = {}
    if "train" in dataset_dict:
        dataset_module["train_dataset"] = dataset_dict["train"]

    if "validation" in dataset_dict:
        dataset_module["eval_dataset"] = dataset_dict["validation"]

    if tokenize:
        if tokenizer_function is None:
            raise ValueError("Please pass a valid tokenizer function.")

        dataset_module["train_dataset"] = dataset_module["train_dataset"].map(tokenizer_function,
                                              batched=True, num_proc=32, remove_columns=["text"])

        dataset_module["eval_dataset"] = dataset_module["eval_dataset"].map(tokenizer_function,
                                              batched=True, num_proc=32, remove_columns=["text"])
    return dataset_module

def _get_merged_dataset(
    dataset_list: List[DatasetArgs],
    data_args: DataArgs,
    tokenizer: PreTrainedTokenizer
) -> Optional[Union["Dataset", "IterableDataset"]]:
    r"""
    Gets the merged datasets in the standard format.
    """
    datasets = []
    for dataset_attr in dataset_list:
        datasets.append(_load_dataset(dataset_attr, data_args, tokenizer))

    return merge_dataset(datasets, data_args, data_args.seed)

def merge_dataset(all_datasets: List[Union["Dataset", "IterableDataset"]], data_args: "DataArgs"
                  , seed) -> Union["Dataset", "IterableDataset"]:
    r"""
    Merges multiple datasets to a unified dataset.
    """
    if len(all_datasets) == 1:
        return all_datasets[0]
    elif data_args.mix_strategy == "concat":
        if data_args.streaming:
            logger.warning("The samples between different datasets will not be mixed in streaming mode.")
        return concatenate_datasets(all_datasets)
    elif data_args.mix_strategy.startswith("interleave"):
        if not data_args.streaming:
            logger.warning("We recommend using `mix_strategy=concat` in non-streaming mode.")

        return interleave_datasets(
            datasets=all_datasets,
            probabilities=data_args.interleave_probs,
            seed=seed,
            stopping_strategy="first_exhausted" if data_args.mix_strategy.endswith("under") else "all_exhausted",
        )
    else:
        raise ValueError("Unknown mixing strategy: {}.".format(data_args.mix_strategy))



