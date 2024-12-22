from random import randint

import torch
from datasets import Dataset
from trl import SFTConfig
from accelerate import Accelerator

import s3_utils
from distillflow.common import get_current_device
from distillflow.model.loader import load_model, load_tokenizer
from distillflow.model.args import ModelArguments
from distillflow.model.finetuning_args import FinetuningArguments
from distillflow.distill_datasets.loader import get_dataset
from distillflow.distill_datasets.dataset_args import DatasetArgs
from distillflow.distill_datasets.template import Alpaca, AlpacaArgs, ShareGpt, ShareGptArgs
from distillflow.trainer.logits_distillation import LogitsTrainer
from distillflow.distill_datasets.dataset_args import DataArgs
from distillflow.trainer.attention_distillation import AttentionTrainer
from distillflow.trainer.layers_distillation import LayersTrainer
from distillflow.distill_datasets.template import ShareGpt, ShareGptArgs, AlpacaArgs, Alpaca
from distillflow.trainer.logits_distillation import LogitsTrainer

def main():
    student_model_args = ModelArguments(
        # model_name_or_path="HuggingFaceTB/SmolLM2-135M-Instruct",#"meta-llama/Llama-3.2-1B-Instruct",
        model_name_or_path="Qwen/Qwen2-0.5B",#"meta-llama/Llama-3.2-1B-Instruct",
        # flash_attn="fa2",
        use_unsloth=False,
        output_attentions=True,
        # enable_liger_kernel=True
        # quantization_bit=8,
        # quantization_method="gptq"
    )
    student_model = load_model(student_model_args, finetuning_args=FinetuningArguments(), is_trainable=True)
    print(student_model)
    teacher_model_args = ModelArguments(
        # model_name_or_path="HuggingFaceTB/SmolLM2-360M-Instruct",#"meta-llama/Llama-3.2-1B-Instruct",
        model_name_or_path="Qwen/Qwen2-1.5B",#"meta-llama/Llama-3.2-1B-Instruct",
        # flash_attn="fa2",
        # quantization_bit=8,
        use_unsloth=False,
        output_attentions=True,
        # enable_liger_kernel=True
        # quantization_bit=8,
        # quantization_method="gptq"
    )
    teacher_model = load_model(teacher_model_args, finetuning_args=FinetuningArguments(), is_trainable=False)

    data_args = DataArgs(
        seed=0,
        train_datasets=[
            DatasetArgs(path="mlabonne/FineTome-100k", dataset_text_field="text", seed=42, template=ShareGpt()),
            DatasetArgs(path="databricks/databricks-dolly-15k",dataset_text_field="text", seed=42, template=Alpaca(args=AlpacaArgs(prompt="instruction", query="context", response="response"))),
        ],
        test_size=1000,
        streaming=False,
    )

    tokenizer = load_tokenizer(student_model_args)["tokenizer"]
    dataset_module = get_dataset(data_args, tokenizer)

    # # ################# for now just load dataset from hugging face.
    # # from datasets import load_dataset
    # # dataset = load_dataset("mlabonne/FineTome-100k", split='train')
    # # dataset = dataset.shuffle(seed=42)
    # #
    # # def sharegpt_format(example):
    # #     conversations = example['conversations']
    # #     message = []
    # #
    # #     if isinstance(conversations, list):
    # #         for conversation in conversations:
    # #             if isinstance(conversation, dict):
    # #                 if conversation.get('from') == 'human':
    # #                     message.append({"role": "user", "content": conversation.get('value', '')})
    # #                 elif conversation.get('from') == 'gpt':
    # #                     message.append({"role": "assistant", "content": conversation.get('value', '')})
    # #                 elif conversation.get('from') == 'system':
    # #                     message.insert(0, {"role": "system", "content": conversation.get('value', '')})
    # #
    # #     if not any(msg.get('role') == 'system' for msg in message):
    # #         message.insert(0, {"role": "system", "content": "You are a helpful assistant."})
    # #
    # #     text = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    # #     return {"text": text}
    # #
    # # tokenizer.chat_template = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    # #
    # # print("Preprocessing and tokenizing dataset...")
    # # original_columns = dataset.column_names
    # # dataset = dataset.map(sharegpt_format, remove_columns=original_columns)
    # #
    # # def tokenize_function(examples):
    # #     return tokenizer(examples["text"], truncation=True, max_length=4096,
    # #                              padding="max_length")
    # #
    # # tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=8, remove_columns=["text"])
    # # tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)
    #
    # dataset_module = {}
    # dataset_module['train_dataset'] = tokenized_dataset['train']
    # dataset_module['eval_dataset'] = tokenized_dataset['test']

    trainer = logits_distill(teacher_model, student_model, dataset_module, tokenizer, data_args)

    # trainer = layers_distill(teacher_model, student_model, dataset_module, tokenizer, data_args)
    # trainer = attention_distill(teacher_model, student_model, dataset_module, tokenizer, data_args)

    device = get_current_device()
    if device.type != "mps":
        accelerator = Accelerator()
        trainer = accelerator.prepare(trainer)

    trainer_stats = trainer.train()

    # upload to s3

    try:
        s3_utils.upload_to_s3('distillflow-output', 'src/results')
    except Exception as e:
        print("received exception while uploading results", e)


def attention_distill(teacher_model, student_model, dataset_module, tokenizer, data_args):
    config = {
        "output_dir": "./results",
        "num_train_epochs": 3,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "save_steps": 1000,
        # "max_steps": 15000, # need to specify with streaming enabled
        "logging_steps": 1,
        "learning_rate": 2e-5,
        "weight_decay": 0.05,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "cosine",
        "resume_from_checkpoint": None,  # Set to a path or True to resume from the latest checkpoint
        "fp16": False,
        "bf16": True,
        "max_grad_norm": 1.0,
        "group_by_length": False
    }
    return AttentionTrainer(
        model=student_model,
        args=SFTConfig(**config),
        dataset_module=dataset_module,
        tokenizer=tokenizer,
        max_seq_length=1024,
        dataset_text_field=data_args.train_dataset.dataset_text_field,
        # Distillation specific arguments
        teacher_model=teacher_model,
        distillation_args={"temperature": 2.0, "alpha": 0.5},
        tokenizer_args={"max_length": 1024,
                        "chat_template": "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
                        }
    )

def layers_distill(teacher_model, student_model, dataset_module, tokenizer, data_args):
    config = {
        "output_dir": "./results",
        "num_train_epochs": 3,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "save_steps": 1000,
        # "max_steps": 15000, # need to specify with streaming enabled
        "logging_steps": 1,
        "learning_rate": 2e-5,
        "weight_decay": 0.05,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "cosine",
        "resume_from_checkpoint": None,  # Set to a path or True to resume from the latest checkpoint
        "fp16": False,
        "bf16": True,
        "max_grad_norm": 1.0,
        "group_by_length": False
    }
    return LayersTrainer(
        model=student_model,
        args=SFTConfig(**config),
        dataset_module=dataset_module,
        tokenizer=tokenizer,
        max_seq_length=1024,
        dataset_text_field=data_args.train_dataset.dataset_text_field,
        # Distillation specific arguments
        teacher_model=teacher_model,
        distillation_args= {"temperature": 2.0, "alpha": 0.5},
        tokenizer_args={"max_length": 1024,
                        "chat_template": "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
                        }
    )

def logits_distill(teacher_model, student_model, dataset_module, tokenizer, data_args):
    config = {
        "output_dir": "./results",
        "num_train_epochs": 3,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "save_steps": 1000,
        # "max_steps": 5000, # need to specify with streaming enabled
        "logging_steps": 1,
        "learning_rate": 2e-5,
        "weight_decay": 0.05,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "cosine",
        "resume_from_checkpoint": None,  # Set to a path or True to resume from the latest checkpoint
        "fp16": False,
        "bf16": True
    }
    return LogitsTrainer(
        model=student_model,
        args=SFTConfig(**config),
        dataset_module=dataset_module,
        tokenizer=tokenizer,
        max_seq_length=4096,
        dataset_text_field="text",
        # Distillation specific arguments
        teacher_model=teacher_model,
        distillation_args= {"temperature": 2.0, "alpha": 0.5},
        tokenizer_args={"max_length": 1024,
                        "chat_template": "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
                        }
    )


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
