from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from .student import Student
from peft import LoftQConfig, LoraConfig, get_peft_model


class Llama3(Student):
    def __init__(self, model_path=None, model_name='unsloth/Llama-3.2-1B-Instruct'):
        super().__init__()
        self.model_path = model_path
        self.model_name= model_name

        if self.model_path:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path,
                                                              torch_dtype="auto",
                                                              device_map="auto"
                                                              )

        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map="auto"
            )


        self.lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1,
                                      target_modules=["q_proj", "v_proj"],
                                      bias='none')

        self.model = get_peft_model(self.model, self.lora_config)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

        ### Instruction:
        {}

        ### Input:
        {}

        ### Response:
        {}"""


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
        texts = []
        EOS_TOKEN = self.tokenizer.eos_token
        for instruction, input, output in zip(instructions, inputs, responses):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            text = self.prompt.format(instruction, input, output) + EOS_TOKEN
            texts.append(text)

        return {"text": texts, }


    def inference(self, query):

        # query has two parts instruction, input
        self.model.eval()

        inputs = self.tokenizer(
            [
                self.prompt.format(
                    query['prompt'],  # instruction
                    query['context'],  # input
                    "",  # output - leave this blank for generation!
                )
            ], return_tensors="pt")

        output = self.model.generate(
            **inputs,
            max_new_tokens=512,  # Maximum length of generated sequence
            # num_return_sequences=1,  # Number of sequences to generate
            # no_repeat_ngram_size=2,  # Avoid repetition
            # top_k=50,  # Top-k sampling
            # top_p=0.95,  # Top-p (nucleus) sampling
            # temperature=0.7,  # Sampling temperature
        )

        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

        return generated_text

