import pandas as pd

from distillflow.distill import Distiller
from distillflow.teacher import TeacherModel
from distillflow.distill_datasets import FlowDataset, FlowData
from datasets import Dataset

class DistillFlow:
    """
    Main class to handle the distillation pipeline using Accelerate.
    """
    def __init__(self, teacher_model: TeacherModel, distiller: Distiller, distill_dataset: FlowDataset):
        self.teacher_model = teacher_model
        self.distiller = distiller
        self.dataset = distill_dataset
        self.train_dataset: FlowData
        self.test_dataset: FlowData

    def prepare_data(self):
        self.train_dataset, self.test_dataset = self.dataset.prepare_data()

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
        print("Loading collected responses...")
        df = pd.read_csv(data_file)
        dataset = Dataset.from_pandas(df)
        print(f"Dataset is: {dataset}")

        self.distiller.fine_tune(dataset=dataset, output_dir=output_dir)
        print(f"Student model fine-tuned and saved to {output_dir}")