from typing import TypedDict, Optional

from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import PreTrainedTokenizer, ProcessorMixin

def load_dataset():
    dataset = load_dataset(
        path=data_path,
        name=data_name,
        data_dir=data_dir,
        data_files=data_files,
        split=dataset_attr.split,
        cache_dir=model_args.cache_dir,
        token=model_args.hf_hub_token,
        streaming=(data_args.streaming and (dataset_attr.load_from != "file")),
        trust_remote_code=True,
    )

    if data_args.streaming and (dataset_attr.load_from == "file"):  # faster than specifying streaming=True
        dataset = dataset.to_iterable_dataset()  # TODO: add num shards parameter

    if dataset_attr.num_samples is not None and not data_args.streaming:
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

    return align_dataset(dataset, dataset_attr, data_args, training_args)