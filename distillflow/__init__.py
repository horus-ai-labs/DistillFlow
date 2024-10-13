import pandas as pd
from distillflow.student.sft import SFTStudent
from distillflow.teacher import TeacherModel
from distillflow.distill_datasets import FlowDataset, FlowData
from datasets import Dataset

class DistillFlow:
    """
    Main class to handle the distillation pipeline using Accelerate.
    """
    def __init__(self, teacher_model: TeacherModel, student_model: SFTStudent, distill_dataset: FlowDataset):
        self.teacher_model = teacher_model
        self.student_model = student_model
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
        prompts = self.train_dataset.get_prompts
        responses = self.test_dataset.get_responses
        # for i, prompt in enumerate(prompts):
        #     if i > 1:
        #         break
        #     print(f"Prompt: {prompt}")
        #
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

        print(f"Fine-tuning student model {self.student_model.model_name}...")
        self.student_model.fine_tune(train_dataset=dataset, output_dir=output_dir)
        print(f"Student model fine-tuned and saved to {output_dir}")