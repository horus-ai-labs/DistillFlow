from random import randint

from datasets import Dataset
from trl import SFTConfig

from distillflow.model.loader import load_model, load_tokenizer
from distillflow.model.args import ModelArguments
from distillflow.model.finetuning_args import FinetuningArguments
from distillflow.distill_datasets.loader import get_dataset, DatasetModule
from distillflow.distill_datasets.dataset_args import DatasetArgs
from distillflow.distill_datasets.template import Alpaca, AlpacaArgs
from distillflow.trainer.logits_distillation import LogitsTrainer
from distillflow.distill_datasets.dataset_args import DataArgs


def main():
    student_model_args = ModelArguments(
        model_name_or_path="arcee-ai/Arcee-Spark",#"meta-llama/Llama-3.2-1B-Instruct",
        # quantization_bit=8,
        # quantization_method="gptq"
    )
    student_model = load_model(student_model_args, finetuning_args=FinetuningArguments(), is_trainable=True)
    teacher_model_args = ModelArguments(
        model_name_or_path="Qwen/Qwen2-1.5B",#"meta-llama/Llama-3.2-1B-Instruct",
        # quantization_bit=8,
        # quantization_method="gptq"
    )
    teacher_model = load_model(teacher_model_args, finetuning_args=FinetuningArguments(), is_trainable=False)

    data_args=DataArgs(
        template=Alpaca(args=AlpacaArgs(
            prompt="instruction",
            query="context",
            response="response"
        )),
        train_dataset=DatasetArgs(path="databricks/databricks-dolly-15k", to_text=True, seed=42),
        test_size=1000,
        streaming=True)

    tokenizer = load_tokenizer(student_model_args)["tokenizer"]
    dataset_module = get_dataset(data_args, tokenizer)

    config = {
        "output_dir": "./results",
        "num_train_epochs": 3,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "save_steps": 1000,
        "max_steps": 5000, # need to specify with streaming enabled
        "logging_steps": 1,
        "learning_rate": 2e-5,
        "weight_decay": 0.05,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "cosine",
        "resume_from_checkpoint": None,  # Set to a path or True to resume from the latest checkpoint
        "fp16": False,
        "bf16": True
    }
    trainer = LogitsTrainer(
        model=student_model,
        args=SFTConfig(**config),
        dataset_module=dataset_module,
        tokenizer=tokenizer,
        max_seq_length=1024,
        dataset_text_field="text",
        # Distillation specific arguments
        teacher_model=teacher_model,
        distillation_args= {"temperature": 2.0, "alpha": 0.5},
        tokenizer_args={"max_length": 1024,
                        "chat_template": "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
                        }
    )
    trainer_stats = trainer.train()


def print_random(dataset: Dataset):
    # print random rows
    random_rows = [randint(0, dataset.num_rows-1) for _ in range(1)]
    sample = dataset.select(random_rows)
    print(sample.to_dict())

# tokenizer_module = load_tokenizer(model_args)
    # tokenizer = tokenizer_module["tokenizer"]
    # dataset_module = get_dataset(template, model_args, data_args, training_args, stage="rm", **tokenizer_module)
    # model = load_model(tokenizer, model_args, finetuning_args=FinetuningArguments(), is_trainable=True)

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
