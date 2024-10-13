class FlowData:
    def __init__(self, data):
        self.data = data

    def get_contexts(self):
        raise NotImplementedError("Data must implement the get_context method.")

    def get_prompts(self):
        raise NotImplementedError("Data must implement the get_prompt method.")

    def get_responses(self):
        raise NotImplementedError("Data must implement the get_response method.")

class FlowDataset:
    def __init__(self):
        self.train_data: FlowData
        self.test_data: FlowData

    def prepare_data(self) -> (FlowData, FlowData):
        raise NotImplementedError("Distill Dataset must implement the prepare_data method.")
