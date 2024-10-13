import sys

sys.path.append('../')
from distillflow import DistillFlow
from distillflow.teacher.anthropic import AnthropicTeacher
from distillflow.student.sft import SFTStudent
from distillflow.distill_datasets.dolly import Dolly
from distillflow.evaluation.rouge import compute_rouge_scores

def main():
    teacher_model = AnthropicTeacher()  # Or use any lightweight teacher model.
    student_model = SFTStudent(model_name='distilgpt2')  # Lightweight model for fast testing.
    dataset = Dolly()

    pipeline = DistillFlow(teacher_model=teacher_model, student_model=student_model, train_dataset=dataset)

    pipeline.prepare_data()
    pipeline.collect_responses(output_file="anthropic_responses.csv")

    pipeline.train_student_model(data_file="anthropic_responses.csv", output_dir="./sft_output_test")

    # validation outputs



    compute_rouge_scores(reference_file, generated_file)

if __name__ == "__main__":
    main()
