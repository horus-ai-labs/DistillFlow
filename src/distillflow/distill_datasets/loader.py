from functools import partial
from typing import TypedDict, Optional, Dict, Any

import numpy as np
from datasets import DatasetDict, load_dataset, Dataset
from transformers import PreTrainedTokenizer

from .dataset_args import DatasetArgs, DataArgs
from .template.template import Template
from ..misc import get_logger
from ..model.args import ModelArguments

logger = get_logger(__name__)

class DatasetModule(TypedDict):
    train_dataset: Optional[Dataset]
    eval_dataset: Optional[Dataset]


def _load_dataset(
        dataset_attr: DatasetArgs,
        data_args: DataArgs,
        tokenizer: PreTrainedTokenizer) -> Dataset:

    if dataset_attr is not None:
        logger.info("Loading dataset {}...".format(dataset_attr))
        dataset = load_dataset(
            path=dataset_attr.path,
            split=dataset_attr.split,
            cache_dir=data_args.cache_dir,
            token=data_args.hf_hub_token,
            streaming=data_args.streaming, # and (dataset_attr.load_from != "file")),
            trust_remote_code=True,
        )
        # Shuffle dataset with a pre-seed
        dataset = dataset.shuffle(seed=dataset_attr.seed)

        # if data_args.streaming and (dataset_attr.load_from == "file"):  # faster than specifying streaming=True
        #     dataset = dataset.to_iterable_dataset()  # TODO: add num shards parameter

        if dataset_attr.num_samples is not None:# and not data_args.streaming:
            target_num = dataset_attr.num_samples
            indexes = np.random.permutation(len(dataset))[:target_num]  # all samples should be included
            target_num -= len(indexes)
            if target_num > 0:
                expand_indexes = np.random.choice(len(dataset), target_num)
                indexes = np.concatenate((indexes, expand_indexes), axis=0)

            assert len(indexes) == dataset_attr.num_samples, "Sample num mismatched."
            dataset = dataset.select(indexes)
            logger.info("Sampled {} examples from dataset {}.".format(dataset_attr.num_samples, dataset_attr))

        if data_args.max_samples is not None:  # truncate dataset
            max_samples = min(data_args.max_samples, len(dataset))
            dataset = dataset.select(range(max_samples))

        column_names = list(next(iter(dataset)).keys())

        dataset = dataset.map(
            partial(data_args.template.convert),
            batched=False,
            remove_columns=column_names,
        )

        if dataset_attr.to_text:
            dataset = dataset.map(partial(to_text, tokenizer), batched=False)

        return dataset

# def _get_merged_dataset(
#         dataset_names: Optional[Sequence[str]],
#         model_args: ModelArguments,
#         # data_args: "DataArguments",
#         training_args: "Seq2SeqTrainingArguments",
#         # stage: Literal["pt", "sft", "rm", "ppo", "kto"],
# ) -> Dataset:
#     r"""
#     Gets the merged datasets in the standard format.
#     """
#     if dataset_names is None:
#         return None
#
#     datasets = []
#     for dataset_attr in get_dataset_list(dataset_names, data_args.dataset_dir):
#         # if (stage == "rm" and dataset_attr.ranking is False) or (stage != "rm" and dataset_attr.ranking is True):
#         #     raise ValueError("The dataset is not applicable in the current training stage.")
#
#         # datasets.append(_load_single_dataset(dataset_attr, model_args, data_args, training_args))
#
#     return merge_dataset(datasets, data_args, seed=training_args.seed)

def to_text(tokenizer: PreTrainedTokenizer, example: Dict[str, Any]) -> Dict[str, Any]:
    system = example["_system"]
    prompt = example["_prompt"]
    response = example["_response"]
    message = [system, prompt, response]
    return {"text": tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)}

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
        test_size = int(data_args.test_size) if data_args.test_size > 1 else data_args.test_size
        dataset = dataset.train_test_split(test_size=test_size, seed=seed)
        return DatasetDict({"train": dataset["train"], "validation": dataset["test"]})

def get_dataset(data_args: DataArgs,
                tokenizer: PreTrainedTokenizer) -> DatasetModule:
    # Load and preprocess dataset
    # with training_args.main_process_first(desc="load dataset"):
    dataset = _load_dataset(data_args.train_dataset, data_args, tokenizer)
    eval_dataset = None
    if data_args.eval_dataset is not None:
        eval_dataset = _load_dataset(data_args.eval_dataset, data_args, tokenizer)

    # with training_args.main_process_first(desc="pre-process dataset"):
    #     dataset = _get_preprocessed_dataset(
    #         dataset, data_args, training_args, stage, template, tokenizer, processor, is_eval=False
    #     )
    #     eval_dataset = _get_preprocessed_dataset(
    #         eval_dataset, data_args, training_args, stage, template, tokenizer, processor, is_eval=True
    #     )
    #
    #     if data_args.val_size > 1e-6:
    #         dataset_dict = split_dataset(dataset, data_args, seed=training_args.seed)
    #     else:
    dataset_dict = {}

    if eval_dataset is None:
        dataset_dict = split_dataset(dataset, data_args, data_args.train_dataset.seed)
    else:
        if data_args.streaming:
            eval_dataset = eval_dataset.shuffle(buffer_size=data_args.buffer_size, seed=data_args.eval_dataset.seed)

        dataset_dict["validation"] = eval_dataset

        if dataset is not None:
            if data_args.streaming:
                dataset = dataset.shuffle(buffer_size=data_args.buffer_size, seed=data_args.train_dataset.seed)

            dataset_dict["train"] = dataset

        dataset_dict = DatasetDict(dataset_dict)

        # if data_args.tokenized_path is not None:
        #     if training_args.should_save:
        #         dataset_dict.save_to_disk(data_args.tokenized_path)
        #         logger.info("Tokenized dataset saved at {}.".format(data_args.tokenized_path))
        #         logger.info("Please restart the training with `tokenized_path: {}`.".format(data_args.tokenized_path))
        #
        #     sys.exit(0)

    dataset_module = {}
    if "train" in dataset_dict:
        dataset_module["train_dataset"] = dataset_dict["train"]

    if "validation" in dataset_dict:
        dataset_module["eval_dataset"] = dataset_dict["validation"]

    return dataset_module