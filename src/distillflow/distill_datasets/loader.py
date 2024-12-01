import os
from functools import partial
from typing import TypedDict, Optional, Sequence, List, Dict, Any

import numpy as np
from datasets import DatasetDict, load_dataset, load_from_disk, Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin, Seq2SeqTrainingArguments

from .dataset_args import DatasetArgs, DataArgs
from .template.template import Template
from ..misc import get_logger
from ..model.args import ModelArguments

logger = get_logger(__name__)

class DatasetModule(TypedDict):
    train_dataset: Optional[Dataset]
    eval_dataset: Optional[Dataset]


def _load_single_dataset(
        dataset_attr: DatasetArgs,
        model_args: ModelArguments,
        tokenizer: PreTrainedTokenizer,
        # data_args: DataArguments,
        template: Template,
        # training_args: Seq2SeqTrainingArguments,
) -> Dataset:
    r"""
    Loads a single dataset and aligns it to the standard format.
    """

    if dataset_attr is None:
        return None
    logger.info("Loading dataset {}...".format(dataset_attr))
    # data_path, data_name, data_dir, data_files = None, None, None, None
    # if dataset_attr.load_from in ["hf_hub", "ms_hub", "om_hub"]:
    #     data_path = dataset_attr.dataset_name
    #     data_name = dataset_attr.subset
    #     data_dir = dataset_attr.folder

    # elif dataset_attr.load_from == "script":
    #     data_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
    #     data_name = dataset_attr.subset
    #     data_dir = dataset_attr.folder

    # elif dataset_attr.load_from == "file":
    #     data_files = []
    #     local_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
    #     if os.path.isdir(local_path):  # is directory
    #         for file_name in os.listdir(local_path):
    #             data_files.append(os.path.join(local_path, file_name))
    #             if data_path is None:
    #                 data_path = FILEEXT2TYPE.get(file_name.split(".")[-1], None)
    #             elif data_path != FILEEXT2TYPE.get(file_name.split(".")[-1], None):
    #                 raise ValueError("File types should be identical.")
    #     elif os.path.isfile(local_path):  # is file
    #         data_files.append(local_path)
    #         data_path = FILEEXT2TYPE.get(local_path.split(".")[-1], None)
    #     else:
    #         raise ValueError("File {} not found.".format(local_path))
    #
    #     if data_path is None:
    #         raise ValueError("Allowed file types: {}.".format(",".join(FILEEXT2TYPE.keys())))
    # else:
    #     raise NotImplementedError("Unknown load type: {}.".format(dataset_attr.load_from))

    # if dataset_attr.load_from == "ms_hub":
    #     require_version("modelscope>=1.11.0", "To fix: pip install modelscope>=1.11.0")
    #     from modelscope import MsDataset
    #     from modelscope.utils.config_ds import MS_DATASETS_CACHE
    #
    #     cache_dir = model_args.cache_dir or MS_DATASETS_CACHE
    #     dataset = MsDataset.load(
    #         dataset_name=data_path,
    #         subset_name=data_name,
    #         data_dir=data_dir,
    #         data_files=data_files,
    #         split=dataset_attr.split,
    #         cache_dir=cache_dir,
    #         token=model_args.ms_hub_token,
    #         use_streaming=(data_args.streaming and (dataset_attr.load_from != "file")),
    #     )
    #     if isinstance(dataset, MsDataset):
    #         dataset = dataset.to_hf_dataset()
    #
    # elif dataset_attr.load_from == "om_hub":
    #     require_version("openmind>=0.8.0", "To fix: pip install openmind>=0.8.0")
    #     from openmind import OmDataset
    #     from openmind.utils.hub import OM_DATASETS_CACHE
    #
    #     cache_dir = model_args.cache_dir or OM_DATASETS_CACHE
    #     dataset = OmDataset.load_dataset(
    #         path=data_path,
    #         name=data_name,
    #         data_dir=data_dir,
    #         data_files=data_files,
    #         split=dataset_attr.split,
    #         cache_dir=cache_dir,
    #         token=model_args.om_hub_token,
    #         streaming=(data_args.streaming and (dataset_attr.load_from != "file")),
    #     )
    # else:
    dataset = load_dataset(
        path=dataset_attr.path,
        # name=data_name,
        # data_dir=data_dir,
        # data_files=data_files,
        split=dataset_attr.split,
        cache_dir=model_args.cache_dir,
        token=model_args.hf_hub_token,
        # streaming=(data_args.streaming and (dataset_attr.load_from != "file")),
        trust_remote_code=True,
    )

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

    # if data_args.max_samples is not None:  # truncate dataset
    #     max_samples = min(data_args.max_samples, len(dataset))
    #     dataset = dataset.select(range(max_samples))

    column_names = list(next(iter(dataset)).keys())

    dataset = dataset.map(
        partial(template.convert),
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

def split_dataset(
        dataset: Dataset, data_args: DataArgs#, seed: int
) -> "DatasetDict":
    r"""
    Splits the dataset and returns a dataset dict containing train set and validation set.

    Supports both map dataset and iterable dataset.
    """
    # if data_args.streaming:
    #     dataset = dataset.shuffle(buffer_size=data_args.buffer_size, seed=seed)
    #     val_set = dataset.take(int(data_args.val_size))
    #     train_set = dataset.skip(int(data_args.val_size))
    #     return DatasetDict({"train": train_set, "validation": val_set})
    # else:
    val_size = int(data_args.val_size) if data_args.val_size > 1 else data_args.val_size
    dataset = dataset.train_test_split(test_size=val_size) #, seed=seed)
    return DatasetDict({"train": dataset["train"], "validation": dataset["test"]})

def get_dataset(template: Template,
                model_args: ModelArguments,
                data_args: DataArgs,
                # data_args: DataArguments,
                # training_args: Seq2SeqTrainingArguments,
                # stage: Literal["pt", "sft", "rm", "ppo", "kto"],
                tokenizer: PreTrainedTokenizer,
                # processor: Optional[ProcessorMixin] = None,
                ) -> DatasetModule:
    # Load and preprocess dataset
    # with training_args.main_process_first(desc="load dataset"):
    dataset = _load_single_dataset(data_args.train_dataset, model_args, tokenizer, template)
    eval_dataset = None
    if data_args.eval_dataset is not None:
        eval_dataset = _load_single_dataset(data_args.eval_dataset, model_args, tokenizer, template)


        # dataset = _get_merged_dataset(data_args, model_args, data_args, training_args)
        # eval_dataset = _get_merged_dataset(data_args.eval_dataset, model_args, data_args, training_args, stage)

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
        dataset_dict = split_dataset(dataset, data_args)
    else:
        # if data_args.streaming:
        #     eval_dataset = eval_dataset.shuffle(buffer_size=data_args.buffer_size, seed=training_args.seed)
        dataset_dict["validation"] = eval_dataset

        if dataset is not None:
            # if data_args.streaming:
            #     dataset = dataset.shuffle(buffer_size=data_args.buffer_size, seed=training_args.seed)

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