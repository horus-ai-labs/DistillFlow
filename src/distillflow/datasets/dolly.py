from . import DistillDataset
from datasets import load_dataset
from datasets import Dataset

class Dolly(DistillDataset):

    def __rename_keys(self, example):
        return {
            "prompt": example["instruction"],
            "context": example["context"],
            "response": example["response"],
            "category": example["category"],
        }

    def prepare_data(self) -> (Dataset, Dataset) :
        dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
        dataset = dataset.map(self.__rename_keys)
        split_datasets = dataset.train_test_split(test_size=0.2)
        return split_datasets['train'], split_datasets['test']