from distillflow import DistillFlow
from distillflow.teacher.anthropic import AnthropicTeacher
from distillflow.distill_datasets.dolly import Dolly
# from distillflow.evaluation.rouge import compute_rouge_scores
from distillflow.distill.sft import SFTWithoutKD
# from distillflow.distill.sft_native import SFT
from distillflow.student.qwen import Qwen
import json

from distillflow.student.llama3 import Llama3
from distillflow.distill.sft_native import SFT
from distillflow.model.loader import load_model, load_tokenizer
from distillflow.model.args import ModelArguments
from distillflow.model.finetuning_args import FinetuningArguments


def main():
    teacher_model = AnthropicTeacher()  # Or use any lightweight teacher model.
    student_model = Llama3()
    dataset = Dolly()

    model_args = ModelArguments(
        model_name_or_path="meta-llama/Llama-3.2-1B-Instruct",
        # quantization_bit=8,
        # quantization_method="gptq"
    )
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    # dataset_module = get_dataset(template, model_args, data_args, training_args, stage="rm", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args=FinetuningArguments(), is_trainable=True)

    # train_args = dict(
    #     model_name_or_path="unsloth/Llama-3.2-1B-Instruct",
    #
    #     stage= "sft",
    #     do_train= True,
    #     finetuning_type="lora",
    #     lora_target= "all",
    #
    #     dataset="alpaca_en_demo",
    #     dataset_dir="/Users/karankhurana/workspace/DistillFlow/src/data",
    #     template="llama3",
    #     cutoff_len=1024,
    #     max_samples=1000,
    #     overwrite_cache= True,
    #     preprocessing_num_workers= 16,
    #
    #     output_dir= "saves/llama3-8b/lora/sft",
    #     logging_steps=10,
    #     save_steps= 50,
    #     overwrite_output_dir= True,
    #
    #     per_device_train_batch_size= 1,
    #     gradient_accumulation_steps= 8,
    #     learning_rate= 1.0e-4,
    #     num_train_epochs= 1.0,
    #     lr_scheduler_type= "cosine",
    #     warmup_ratio=0.1,
    #     bf16= True,
    #     ddp_timeout= 180000000,
    #
    #     val_size= 0.1,
    #     per_device_eval_batch_size= 1,
    #     eval_strategy= "steps",
    #     eval_steps= 50
    # )
    # run_exp(args=train_args)



    # evaluator = Evaluator(args=evaluator_args)
    # evaluator.eval()

    # distiller = SFT(student_model)
    #
    # pipeline = DistillFlow(teacher_model=teacher_model, distiller=distiller, distill_dataset=dataset)

    # pipeline.prepare_data()
    # pipeline.collect_responses(output_file="anthropic_responses.csv")

    # pipeline.train_student_model(output_dir="./sft_output_test")

    # validation outputs

    # for idx, data in enumerate(pipeline.test_dataset):
    #     if idx>5:
    #         break
    #     # print(data.keys())
    #     print(pipeline.infer(data))

    # compute_rouge_scores(reference_file, generated_file)

if __name__ == "__main__":
    main()
