from . import TrainDataset
from datasets import load_dataset

class Dolly(TrainDataset):

    def __init__(self):
        self.data = None

    def prepare_data(self):
        self.data = load_dataset("databricks/databricks-dolly-15k", split="train")

    def get_prompts(self):
        return [instruction + "\n\n" + context for instruction, context in zip(self.data['instruction'], self.data['context'])]

    def get_responses(self):
        return self.data['response']
