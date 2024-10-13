from . import FlowDataset
from . import FlowData
from datasets import load_dataset

class Dolly(FlowDataset):

    def prepare_data(self) :
        data = load_dataset("databricks/databricks-dolly-15k", split="train")
        split_datasets = data.train_test_split(test_size=0.2)
        return DollyData(split_datasets["train"]), DollyData(split_datasets["test"])

class DollyData(FlowData):
    def get_contexts(self):
        return self.data['context']

    def get_prompts(self):
        return self.data['instruction']

    def get_responses(self):
        return self.data['response']
