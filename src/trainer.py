import yaml
import argparse

from accelerate import Accelerator
from pydantic import ValidationError
from transformers import PreTrainedModel, PreTrainedTokenizer

import s3_utils
from distillflow.config import Config
from distillflow.common import get_current_device
from distillflow.config.validator import print_validation_error
from distillflow.datasets.loader import get_dataset
from distillflow.model.loader import load_model
from distillflow.trainer.attention_distillation import AttentionTrainer
from distillflow.trainer.layers_distillation import LayersTrainer
from distillflow.trainer.logits_distillation import LogitsTrainer

from accelerate.utils import DeepSpeedPlugin

from distillflow.trainer.fine_tuning import FineTuning


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


def prepare_model(model_args, accelerator, accelerator_state, is_trainable) -> (PreTrainedModel, PreTrainedTokenizer):
    """Prepare model for training."""

    model, tokenizer = load_model(model_args, is_trainable=is_trainable)
    if accelerator is not None:
        accelerator.state.select_deepspeed_plugin(accelerator_state)
        model = accelerator.prepare(model)

    return model, tokenizer

def main():
    args = parse_args()
    try:
        config = Config(**(load_config(args.config)))
    except ValidationError as error:
        print_validation_error(error)
        return

    # Handle device
    device = get_current_device()

    accelerator = None
    student_plugin = DeepSpeedPlugin(hf_ds_config=config.student_model.deepspeed_config)
    teacher_plugin = DeepSpeedPlugin(hf_ds_config=config.teacher_model.deepspeed_config) if config.teacher_model else None

    if teacher_plugin:
        deepspeed_plugins = {"student": student_plugin, "teacher": teacher_plugin}
    else:
        deepspeed_plugins = {"student": student_plugin}

    if device.type != "mps":
        accelerator = Accelerator(deepspeed_plugins=deepspeed_plugins)


    # Load student model
    student_model, student_tokenizer = prepare_model(config.student_model, accelerator=accelerator, accelerator_state='student', is_trainable=True)

    teacher_model, teacher_tokenizer = prepare_model(config.teacher_model, accelerator=accelerator, accelerator_state='teacher', is_trainable=False) if config.teacher_model else None, None

    # Load dataset
    def tokenizer_function(examples):
        # print(examples[config.data.text_field])
        # exit()
        return student_tokenizer(examples[config.data.text_field], truncation=True, max_length=config.distill.max_seq_length,
                                 padding="max_length", return_tensors="pt")

    dataset_module = get_dataset(config.data, student_tokenizer, tokenizer_function=tokenizer_function)

    # Initialize trainer
    trainer_class_mapping = {
        "logits": LogitsTrainer,
        "layers": LayersTrainer,
        "attention": AttentionTrainer,
        "fine-tune": FineTuning
    }
    trainer = trainer_class_mapping[config.distill.type](
        accelerator=accelerator,
        distill_args=config.distill,
        teacher_model=teacher_model,
        model=student_model,
        dataset_module=dataset_module,
        tokenizer=student_tokenizer
    )

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


if __name__ == "__main__":
    main()