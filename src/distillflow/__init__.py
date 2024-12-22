import pandas as pd
from transformers import PreTrainedModel

from .distill import Distiller
from datasets import Dataset
from .distill_datasets import DistillDataset

class DistillFlow:
    """
    Main class to handle the distillation pipeline using Accelerate.
    """
    def __init__(self, teacher_model: PreTrainedModel, distiller: Distiller, distill_dataset: DistillDataset):
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
        self.train_response = self.train_dataset
        # self.train_dataset = self.train_dataset.map(self.distiller.student.encode, batched=True)
        # self.test_dataset = self.test_dataset.map(self.distiller.student.encode, batched=True)

    def collect_responses(self):
        """
        Use the teacher model to collect responses for the dataset.
        """
        print("Collecting responses using the teacher model...")
        contexts = self.train_dataset['context']
        prompts = self.train_dataset['prompt']
        responses = self.train_dataset['response']

        for context, prompt in zip(contexts, prompts):
            response = self.teacher_model.generate_response(context, prompt)


    def train_student_model(self, output_dir='./sft_output'):
        """
        Fine-tune the student model using collected responses.
        Args:
            output_dir: Directory to save the fine-tuned model.
        """
        if self.distiller.device == 'cpu':
            print("GPU is not available. Will use CPU...")
        else:
            print("Using GPUs...")
        print("Loading collected responses...")
        print(f"Dataset used for training {self.train_response}")

        # load_model()
        self.distiller.fine_tune(dataset=self.train_response, output_dir=output_dir)
        print(f"Student model fine-tuned and saved to {output_dir}")

    def infer(self, data):

        if isinstance(data, Dataset):
            #iterate on the data and pass it to model. For efficiency, we should batch it.
            # for idx, sample in enumerate(data):
            print("Not implemented yet.")

        elif isinstance(data, dict):
            # need to contain
            print("Not implemented yet.")
            return self.distiller.student.inference(data)
        else:
            raise NotImplementedError("no support for other data types")
        # from transformers import AutoModel, AutoTokenizer
        # # model = AutoModel.from_pretrained(model_directory)
        # # tokenizer = AutoTokenizer.from_pretrained(model_directory)
        # import pandas as pd
        #
        # df = pd.read_csv("/Users/aayushgupta/dev/DistillFlow/examples/anthropic_responses.csv")
        # dataset = Dataset.from_pandas(df)
        #
        # for step, batch in enumerate(dataset):
        #     print(batch)
        #     break
        # print(df.head())
        #
        # model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        # model = AutoModelForCausalLM.from_pretrained(
        #     model_name,
        #     torch_dtype="auto",
        #     device_map="auto"
        # )
        # tokenizer = AutoTokenizer.from_pretrained(model_name)
        #
        # model.eval()
        #
        # # prompt = "What is the difference between a putter and a driver in golf?"
        # prompt = "Imagine you are a mom. Write a talk track for convince your young son, who does not want to stop playing, to leave for school."
        # messages = [
        #     {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        #     {"role": "user", "content": prompt}
        # ]
        # text = tokenizer.apply_chat_template(
        #     messages,
        #     tokenize=False,
        #     add_generation_prompt=True
        # )
        # print(text)
        # model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        #
        # generated_ids = model.generate(
        #     **model_inputs,
        #     max_new_tokens=512
        # )
        # generated_ids = [
        #     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        # ]
        #
        # outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        #
        # print(outputs)
        # return outputs