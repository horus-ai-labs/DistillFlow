import yaml
import argparse

from accelerate import Accelerator
from pydantic import ValidationError
from trl import SFTConfig

import s3_utils
from distillflow.config import Config
from distillflow.common import get_current_device
from distillflow.config.validator import print_validation_error
from distillflow.datasets.loader import get_dataset
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
    try:
        config = Config(**(load_config(args.config)))
    except ValidationError as error:
        print_validation_error(error)
        return

    # Handle device
    device = get_current_device()

    # Load student model
    student_model = load_model(config.student_model, finetuning_args=FinetuningArguments(),
                               is_trainable=True)

    # Load teacher model
    teacher_model = load_model(config.teacher_model, finetuning_args=FinetuningArguments(), is_trainable=False)

    # Load tokenizer and dataset
    tokenizer_template = config.tokenizer.template
    tokenizer = load_tokenizer(config.student_model, template=tokenizer_template)

    def tokenizer_function(examples):
        return tokenizer(examples[config.data.text_field], truncation=True, max_length=config.distill.max_seq_length,
                                 padding="max_length", return_tensors="pt")

    dataset_module = get_dataset(config.data, tokenizer, tokenizer_function=tokenizer_function)

    accelerator = Accelerator()

    # Initialize trainer
    trainer_class_mapping = {
        "logits": LogitsTrainer,
        "layers": LayersTrainer,
        "attention": AttentionTrainer
    }
    trainer = trainer_class_mapping[config.distill.type](
        accelerator=accelerator,
        config=config.distill,
        teacher_model=teacher_model,
        model=student_model,
        dataset_module=dataset_module,
        tokenizer=tokenizer
    )
    if device.type != "mps":
        trainer = accelerator.prepare(trainer)

    # Train model
    trainer_stats = trainer.train(resume_from_checkpoint=config.distill.resume_from_checkpoint)
    print(trainer_stats)
    output_dir = config.distill.sft_config.output_dir
    trainer.save_model(output_dir)

    # upload results to s3
    try:
        s3_utils.upload_to_s3('distillflow-output', f'{output_dir}')
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