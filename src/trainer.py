from functools import partial

import yaml
import argparse

from accelerate import Accelerator
from trl import SFTConfig

import s3_utils
from distillflow.common import get_current_device
from distillflow.datasets.dataset_args import DatasetArgs, DataArgs
from distillflow.datasets.loader import get_dataset
from distillflow.datasets.template import ShareGpt, Alpaca, ShareGptArgs, AlpacaArgs
from distillflow.model.args import ModelArguments
from distillflow.model.finetuning_args import FinetuningArguments
from distillflow.model.loader import load_model, load_tokenizer
from distillflow.trainer.attention_distillation import AttentionTrainer
from distillflow.trainer.layers_distillation import LayersTrainer
from distillflow.trainer.logits_distillation import LogitsTrainer


def load_config(config_path):
    """Load YAML configuration."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run model training with YAML configuration.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config)

    # Handle device
    device = get_current_device()

    # Load student model
    student_model_args = ModelArguments(**config["student_model"])
    student_model = load_model(student_model_args, finetuning_args=FinetuningArguments(),
                               is_trainable=True)

    # Load teacher model
    teacher_model_args = ModelArguments(**config["teacher_model"])
    teacher_model = load_model(teacher_model_args, finetuning_args=FinetuningArguments(), is_trainable=False)

    # Load data arguments
    train_datasets = [
        DatasetArgs(
            template=ShareGpt(ShareGptArgs(**ds.get("template_args", {}))) if ds["template"] == "sharegpt" else Alpaca(AlpacaArgs(**ds.get("template_args", {}))),
            **{k: v for k, v in ds.items() if k not in ["template"]}
        ) for ds in config["data"]["train_datasets"]
    ]

    data_args = DataArgs(
        train_datasets=train_datasets,
        **{k: v for k, v in config["data"].items() if k not in ["train_datasets"]}
    )

    # Load tokenizer and dataset
    tokenizer_template = config.get("tokenizer", {}).get("template", None)
    tokenizer = load_tokenizer(student_model_args, template=tokenizer_template)

    def tokenizer_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=config["distill"]["max_seq_length"],
                                 padding="max_length", return_tensors="pt")

    dataset_module = get_dataset(data_args, tokenizer, tokenizer_function=tokenizer_function)

    accelerator = None
    if device.type != "mps":
        accelerator = Accelerator()

    # Initialize trainer
    distillation_type = config["distill"]["type"]
    distill_config = config["distill"]["sft_config"]
    text_field = config["data"]["text_field"]
    max_seq_length, distillation_args = config["distill"]["max_seq_length"], config["distill"]["distillation_args"]
    trainer = None
    if distillation_type == "logits":
        trainer = logits_distill(accelerator, distill_config, teacher_model, student_model, dataset_module, tokenizer, text_field, max_seq_length, distillation_args)
    elif distillation_type == "layers":
        trainer = layers_distill(distill_config, teacher_model, student_model, dataset_module, tokenizer, text_field, max_seq_length, distillation_args)
    elif distillation_type == "attention":
        trainer = attention_distill(distill_config, teacher_model, student_model, dataset_module, tokenizer, text_field, max_seq_length, distillation_args)

    if device.type != "mps":
        trainer = accelerator.prepare(trainer)

    # Train model
    trainer_stats = trainer.train(resume_from_checkpoint=config["distill"]["sft_config"]["resume_from_checkpoint"])
    print(trainer_stats)
    trainer.save_model(config["distill"]["sft_config"]["output_dir"])

    # upload results to s3
    try:
        s3_utils.upload_to_s3('distillflow-output', f'src/{distill_config["output_dir"]}')
    except Exception as e:
        print("received exception while uploading results", e)

def tokenize_function(tokenizer, max_length, text_field, examples):
    return tokenizer(examples[text_field], truncation=True, max_length=max_length,
                     padding="max_length")


def attention_distill(config, teacher_model, student_model, dataset_module, tokenizer, text_field, max_seq_length, distillation_args):
    return AttentionTrainer(
        model=student_model,
        args=SFTConfig(**config),
        dataset_module=dataset_module,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        dataset_text_field=text_field,
        # Distillation specific arguments
        teacher_model=teacher_model,
        distillation_args=distillation_args,
    )

def layers_distill(config, teacher_model, student_model, dataset_module, tokenizer, text_field, max_seq_length, distillation_args):
    return LayersTrainer(
        model=student_model,
        args=SFTConfig(**config),
        dataset_module=dataset_module,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        dataset_text_field=text_field,
        # Distillation specific arguments
        teacher_model=teacher_model,
        distillation_args= distillation_args
    )

def logits_distill(accelerator, config, teacher_model, student_model, dataset_module, tokenizer, text_field, max_seq_length, distillation_args):
    return LogitsTrainer(
        accelerator=accelerator,
        model=student_model,
        args=SFTConfig(**config),
        dataset_module=dataset_module,
        tokenizer=None,
        max_seq_length=max_seq_length,
        dataset_text_field=text_field,
        # Distillation specific arguments
        teacher_model=teacher_model,
        distillation_args= distillation_args
    )

if __name__ == "__main__":
    main()