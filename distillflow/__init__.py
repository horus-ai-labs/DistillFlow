import pandas as pd

from distillflow.distill import Distiller
from distillflow.teacher import TeacherModel
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

class DistillFlow:
    """
    Main class to handle the distillation pipeline using Accelerate.
    """
    def __init__(self, teacher_model: TeacherModel, distiller: Distiller, distill_dataset: Dataset):
        self.teacher_model = teacher_model
        self.distiller = distiller
        self.dataset = distill_dataset
        # self.train_dataset: FlowData
        # self.test_dataset: FlowData

        # call it here. Write code to support and merge multiple datasets.
        self.prepare_data()

    def prepare_data(self):
        self.train_dataset, self.test_dataset = self.dataset.prepare_data()
        self.train_dataset = self.train_dataset.map(self.distiller.student.encode, batched=True)
        self.test_dataset = self.test_dataset.map(self.distiller.student.encode, batched=True)


    def collect_responses(self, output_file="responses.csv"):
        """
        Use the teacher model to collect responses for the dataset.
        Args:
            output_file: CSV file to save the prompts and responses.
        """
        print("Collecting responses using the teacher model...")
        responses = []
        contexts = self.train_dataset.get_contexts()
        prompts = self.train_dataset.get_prompts()
        for context, prompt, response in zip(contexts, prompts, self.train_dataset.get_responses()):
            if context is None or context == '':
                context = "No context"

            responses.append({"context": context, "prompt": prompt, "response": response})
        # for i, prompt in enumerate(prompts):
        #     print(f"Generating response for prompt {i+1}/{len(self.train_dataset)}")
        #     response = self.teacher_model.generate_response(prompt)
        #     responses.append({"prompt": prompt, "response": response})

        df = pd.DataFrame(responses)
        df.to_csv(output_file, index=False)
        print(f"Saved responses to {output_file}")
        return output_file

    def train_student_model(self, data_file="responses.csv", output_dir='./sft_output'):
        """
        Fine-tune the student model using collected responses.
        Args:
            data_file: CSV file containing prompts and responses.
            output_dir: Directory to save the fine-tuned model.
        """
        # print("Loading collected responses...")
        # df = pd.read_csv(data_file)
        # dataset = Dataset.from_pandas(df)
        # print(f"Dataset is: {dataset}")

        train_dataset = self.train_dataset

        self.distiller.fine_tune(dataset=train_dataset, output_dir=output_dir)
        print(f"Student model fine-tuned and saved to {output_dir}")

    def infer(self, data_file, model_directory):
        from transformers import AutoModel, AutoTokenizer
        # model = AutoModel.from_pretrained(model_directory)
        # tokenizer = AutoTokenizer.from_pretrained(model_directory)
        import pandas as pd

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
