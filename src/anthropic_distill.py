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
from datasets import load_dataset
from accelerate import Accelerator
from transformers import AutoTokenizer, TrainingArguments
def main():
    config = {
        "project_name": "distil-logits",
        "dataset": {
            "name": "TIGER-Lab/WebInstructSub",
            "split": "train",
            "num_samples": 200000, # You can pass a number here to limit the number of samples to use.
            "seed": 42
        },
        "models": {
            "teacher": "neuralmagic/Llama-3.2-3B-Instruct-FP8",
            "student": "neuralmagic/Llama-3.2-1B-Instruct-FP8"
        },
        "tokenizer": {
            "max_length": 4096,
            "chat_template": "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        },
        "training": {
            "output_dir": "./results",
            "num_train_epochs": 3,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "save_steps": 1000,
            "logging_steps": 1,
            "learning_rate": 2e-5,
            "weight_decay": 0.05,
            "warmup_ratio": 0.1,
            "lr_scheduler_type": "cosine",
            "resume_from_checkpoint": None,  # Set to a path or True to resume from the latest checkpoint
            "fp16": True,
            "bf16": False
        },
        "distillation": {
            "temperature": 2.0,
            "alpha": 0.5
        },
        "model_config": {
            "use_flash_attention": True
        }
        # "spectrum": {
        #     "layers_to_unfreeze": "/workspace/spectrum/snr_results_Qwen-Qwen2-1.5B_unfrozenparameters_50percent.yaml" # You can pass a spectrum yaml file here to freeze layers identified by spectrum.
        # }
    }

    model_args = ModelArguments(
        model_name_or_path=config["models"]["teacher"],
        # use_unsloth=True
    )
    # tokenizer_module = load_tokenizer(model_args)
    teacher_tokenizer = AutoTokenizer.from_pretrained(config["models"]["teacher"])
    # dataset_module = get_dataset(template, model_args, data_args, training_args, stage="rm", **tokenizer_module)
    teacher_model = load_model(teacher_tokenizer, model_args, finetuning_args=FinetuningArguments(), is_trainable=False)

    model_args = ModelArguments(
        model_name_or_path=config["models"]["student"],
        # use_unsloth=True
    )
    student_tokenizer = AutoTokenizer.from_pretrained(config["models"]["student"])
    # dataset_module = get_dataset(template, model_args, data_args, training_args, stage="rm", **tokenizer_module)
    # by default LORA is happening in finetuning.
    student_model = load_model(student_tokenizer, model_args, finetuning_args=FinetuningArguments(), is_trainable=True)


    accelerator = Accelerator()
    device = accelerator.device

    print(device)

    dataset = load_dataset(config["dataset"]["name"], split=config["dataset"]["split"])
    dataset = dataset.shuffle(seed=config["dataset"]["seed"])

    if "num_samples" in config["dataset"]:
        dataset = dataset.select(range(config["dataset"]["num_samples"]))

    print(dataset[0])

    def sharegpt_format(example):
        question = example['question']
        answer = example['answer']
        message = []

        # if isinstance(conversations, list):
        #     for conversation in conversations:
        #         if isinstance(conversation, dict):
        #             if conversation.get('from') == 'human':
        #                 message.append({"role": "user", "content": conversation.get('value', '')})
        #             elif conversation.get('from') == 'gpt':
        #                 message.append({"role": "assistant", "content": conversation.get('value', '')})
        #             elif conversation.get('from') == 'system':
        #                 message.insert(0, {"role": "system", "content": conversation.get('value', '')})

        # if not any(msg.get('role') == 'system' for msg in message):
        #     message.insert(0, {"role": "system", "content": "You are a helpful assistant."})

        message.append({"role": "system", "content": "You are a helpful assistant"})
        message.append({"role": "user", "content": question})
        message.append({"role": "assistant", "content": answer})

        text = student_tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        return {"text": text}

    print("Preprocessing and tokenizing dataset...")
    original_columns = dataset.column_names
    dataset = dataset.map(sharegpt_format, remove_columns=original_columns)

    print(dataset[0])

    def tokenize_function(examples):
        return student_tokenizer(examples["text"], truncation=True, max_length=config["tokenizer"]["max_length"],
                                 padding="max_length")

    tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=8, remove_columns=["text"])
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)

    training_arguments = TrainingArguments(**config["training"])

    trainer = LogitsTrainer(
        model=student_model,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=student_tokenizer,
        args=training_arguments,
        max_seq_length=config["tokenizer"]["max_length"],
        dataset_text_field="text",
    )

    trainer.teacher_model = teacher_model

    trainer = accelerator.prepare(trainer)

    trainer.train(resume_from_checkpoint=config["training"]["resume_from_checkpoint"])

    trainer.save_model(config["training"]["output_dir"])

if __name__ == "__main__":
    main()
