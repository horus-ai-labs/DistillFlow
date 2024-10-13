import sys

sys.path.append('../')
# sys.path.append('../src')

from distillflow import DistillFlow
from distillflow.teacher.anthropic import AnthropicTeacher
from distillflow.distill_datasets.dolly import Dolly
from distillflow.evaluation.rouge import compute_rouge_scores
from distillflow.student.distillbert import DistillBert
from distillflow.distill.sft import SFTWithoutKD

def main():
    teacher_model = AnthropicTeacher()  # Or use any lightweight teacher model.
    student_model = DistillBert()
    dataset = Dolly()
    distiller = SFTWithoutKD(student_model)

    pipeline = DistillFlow(teacher_model=teacher_model, distiller=distiller, distill_dataset=dataset)

    pipeline.prepare_data()
    pipeline.collect_responses(output_file="anthropic_responses.csv")

    pipeline.train_student_model(data_file="anthropic_responses.csv", output_dir="./sft_output_test")

    # validation outputs



    # compute_rouge_scores(reference_file, generated_file)

if __name__ == "__main__":
    main()
