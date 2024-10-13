# from . import FlowDataset
# from . import FlowData
from datasets import load_dataset
# from datasets import Dataset


class Dolly():

    def __rename_keys(self, example):
        return {
            "prompt": example["instruction"],
            "context": example["context"],
            "response": example["response"],
            "category": example["category"],
        }

    def prepare_data(self) :
        dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
        dataset = dataset.map(self.__rename_keys)
        split_datasets = dataset.train_test_split(test_size=0.2)
        return split_datasets['train'], split_datasets['test']

# class DollyData(FlowData):
#     def get_contexts(self):
#         return self.data['context']
#
#     def get_prompts(self):
#         return self.data['instruction']
#
#     def get_responses(self):
#         return self.data['response']
