from transformers import BatchEncoding

class Student:
    def __init__(self):
        self.model_name = None

    def encode(self, batch) -> BatchEncoding:
        raise NotImplementedError("Student must implement the encode method.")

    def forward_pass(self, batch):
        raise NotImplementedError("Student must implement the encode method.")

