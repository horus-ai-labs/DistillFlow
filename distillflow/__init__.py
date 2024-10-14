import pandas as pd

from distillflow.distill import Distiller
from distillflow.teacher import TeacherModel
from datasets import Dataset
from distillflow.distill_datasets import DistillDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

class DistillFlow:
    """
    Main class to handle the distillation pipeline using Accelerate.
    """
    def __init__(self, teacher_model: TeacherModel, distiller: Distiller, distill_dataset: DistillDataset):
        self.teacher_model = teacher_model
        self.distiller = distiller
        self.dataset = distill_dataset
        self.train_dataset = Dataset.from_dict({})
        self.test_dataset = Dataset.from_dict({})
        self.train_response = Dataset.from_dict({})

        # call it here. Write code to support and merge multiple datasets.
        self.prepare_data()

    def prepare_data(self):
        self.train_dataset, self.test_dataset = self.dataset.prepare_data()
        # For the time being, write to train dataset to train response
        self.collect_responses()

    def collect_responses(self):
        """
        Use the teacher model to collect responses for the dataset.
        """
        print("Collecting responses using the teacher model...")
        contexts = self.train_dataset['context']
        prompts = self.train_dataset['prompt']
        responses = self.train_dataset['response']

        # TODO: use the below response generated from teacher instead
        # for context, prompt in zip(contexts, prompts):
            # response = self.teacher_model.generate_response(context, prompt)
        self.train_response = Dataset.from_dict({
            "prompt": prompts,
            "context": contexts,
            "response": responses,
        })

    def train_student_model(self, output_dir='./sft_output'):
        """
        Fine-tune the student model using collected responses.
        Args:
            output_dir: Directory to save the fine-tuned model.
        """
        if self.distiller.device == 'cpu':
            print("GPU is not available. Will use CPU...")
        print("Loading collected responses...")
        print(f"Dataset used for training {self.train_response}")

        self.distiller.fine_tune(dataset=self.train_response, output_dir=output_dir)
        print(f"Student model fine-tuned and saved to {output_dir}")

    def infer(self, data_file, model_directory):
        # model = AutoModel.from_pretrained(model_directory)
        # tokenizer = AutoTokenizer.from_pretrained(model_directory)

        df = pd.read_csv("/Users/aayushgupta/dev/DistillFlow/examples/anthropic_responses.csv")
        dataset = Dataset.from_pandas(df)

        for step, batch in enumerate(dataset):
            print(batch)
            break
        print(df.head())

        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        model.eval()

        # prompt = "What is the difference between a putter and a driver in golf?"
        prompt = "Imagine you are a mom. Write a talk track for convince your young son, who does not want to stop playing, to leave for school."
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print(outputs)
        # inputs = "What is the difference between a putter and a driver in golf?"

        # text_generator = pipeline(
        #     "question-answering",
        #     model=model,
        #     tokenizer=tokenizer,
        #     # framework="tf",
        # )
        #
        # outputs = text_generator(inputs)
        #
        # print(outputs)
        return outputs

        # tokenizer.pad_token = tokenizer.eos_token
        # model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length",
        #                          return_tensors="pt")
        # print(model_inputs)
        #
        # outputs = model(**model_inputs)
        #
        # logits = outputs.logits
        #
        # # Select the token with the highest score for each position (argmax across vocab size dimension)
        # predicted_token_ids = torch.argmax(logits, dim=-1)
        #
        # # Convert token IDs to text
        # predicted_text = tokenizer.decode(predicted_token_ids[0], skip_special_tokens=True)
        #
        # # Output the predicted text
        # print(predicted_text)

        # for key in outputs:
        #     print(key)

        # return outputs
