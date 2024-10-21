from datasets import Dataset

class DistillDataset:
    def __init__(self):
        self.train_data: Dataset
        self.test_data: Dataset

    def prepare_data(self) -> (Dataset, Dataset):
        raise NotImplementedError("Distill Dataset must implement the prepare_data method.")
