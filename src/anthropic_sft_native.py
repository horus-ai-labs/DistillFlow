import sys

sys.path.append('../')
# sys.path.append('../src')
from torch import mps

from distillflow import DistillFlow
from distillflow.teacher.anthropic import AnthropicTeacher
from distillflow.distill_datasets.dolly import Dolly
from distillflow.evaluation.rouge import compute_rouge_scores
from distillflow.student.qwen import Qwen
from distillflow.distill.sft_native import SFT
import pandas as pd
def main():
    mps.empty_cache()
    teacher_model = AnthropicTeacher()  # Or use any lightweight teacher model.
    student_model = Qwen()
    dataset = Dolly()
    distiller = SFT(student_model)

    pipeline = DistillFlow(teacher_model=teacher_model, distiller=distiller, distill_dataset=dataset)

    # pipeline.prepare_data()
    # pipeline.collect_responses(output_file="anthropic_responses.csv")
    pipeline.train_student_model(output_dir="./sft_native_output_test")
    # validation outputs
    # parse data from test_dataset
    for idx, data in enumerate(pipeline.test_dataset):
        if idx>5:
            break
        print(data.keys())
        print(pipeline.infer(data))


    # compute_rouge_scores(reference_file, generated_file)

if __name__ == "__main__":
    main()
