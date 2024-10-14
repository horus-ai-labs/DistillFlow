from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from distillflow.student import Student


class Qwen(Student):
    def __init__(self, model_name='Qwen/Qwen2.5-0.5B-Instruct'):
        super().__init__()
        self.model_name= model_name
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    # def encode(self, batch):
    #     inputs = batch["prompt"]
    #     targets = batch["response"]
    #     model_inputs = self.tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    #     with self.tokenizer.as_target_tokenizer():
    #         labels = self.tokenizer(targets, max_length=512, truncation=True, padding="max_length")
    #     model_inputs["labels"] = labels["input_ids"]  # Model expects 'labels' for the target sequence.
    #     return model_inputs


    def encode(self, batch):
        inputs = batch["context"]
        instructions = batch["prompt"]
        responses = batch["response"]

        prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

        ### Instruction:
        {}

        ### Input:
        {}

        ### Response:
        {}"""

        texts = []
        EOS_TOKEN = self.tokenizer.eos_token
        for instruction, input, output in zip(instructions, inputs, responses):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            text = prompt.format(instruction, input, output) + EOS_TOKEN
            texts.append(text)

        return {"text": texts, }

    def forward_pass(self, batch):
        return self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            start_positions=batch['start_positions'],
            end_positions=batch['end_positions']
        )
