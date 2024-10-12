class TrainDataset:
    def __init__(self):
        pass

    def prepare_data(self):
        raise NotImplementedError("Datasets must implement the prepare_data method.")

    def get_prompts(self):
        raise NotImplementedError("Datasets must implement the get_prompt method.")

    def get_responses(self):
        raise NotImplementedError("Datasets must implement the get_response method.")