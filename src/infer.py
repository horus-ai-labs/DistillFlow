import yaml
import argparse

from accelerate import Accelerator
from pydantic import ValidationError

import s3_utils
from torch.utils.data import DataLoader
from distillflow.config import Config
from distillflow.common import get_current_device
from distillflow.config.validator import print_validation_error
from distillflow.datasets.loader import get_dataset
from distillflow.model.finetuning_args import FinetuningArguments
from distillflow.model.loader import load_model, load_tokenizer
from distillflow.trainer.attention_distillation import AttentionTrainer
from distillflow.trainer.layers_distillation import LayersTrainer
from distillflow.trainer.logits_distillation import LogitsTrainer

from accelerate.utils import DeepSpeedPlugin, get_active_deepspeed_plugin, is_deepspeed_available

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

def prepare_model(model_config, accelerator, accelerator_state, finetuning_args, is_trainable):
    """Prepare model for training."""

    model = load_model(model_config, finetuning_args=finetuning_args,
                               is_trainable=is_trainable)
    if accelerator is not None:
        accelerator.state.select_deepspeed_plugin(accelerator_state)
        model = accelerator.prepare(model)

    return model


def main():
    args = parse_args()
    try:
        config = Config(**(load_config(args.config)))

    except ValidationError as error:
        print_validation_error(error)
        return


    device = get_current_device()

    # Load tokenizer and dataset
    tokenizer_template = config.tokenizer.template
    # Auto - regressive inference. padding on the left.
    tokenizer = load_tokenizer(config.student_model, template=tokenizer_template, padding_side="left")

    # def tokenizer_function(examples):
    #     return tokenizer(examples[config.data.text_field], truncation=True, max_length=config.distill.max_seq_length,
    #                              padding="max_length", return_tensors="pt")

    # load data and tokenize somehow.
    # Add load+_dataset = "horus-ai-labs/mmlu-sharegpt-all" and run eval.
    prompt = "Question: For T: Z x Z -> Z where T(1, 0) = 3 and T(0, 1) = -5, find T(-3,2).\n\nChoices:\nA) -19\nB) -10\nC) 19\nD) 10"

    messages = [{"role": "user", "content": prompt}]
    tokenizer.eos_token = "<|im_end|>"
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    # Load Single Model (by default student is loaded and teacher is ignored).
    accelerator = None
    student_plugin = DeepSpeedPlugin(hf_ds_config=config.student_model.deepspeed_config)
    deepspeed_plugins = {"student": student_plugin}


    if device.type != "mps":
        accelerator = Accelerator(deepspeed_plugins=deepspeed_plugins)

    student_model = prepare_model(config.student_model, accelerator=accelerator, accelerator_state='student',
                                  finetuning_args=FinetuningArguments(), is_trainable=False)

    if device.type == "mps":
        student_model = student_model.to(device)
        model_inputs = model_inputs.to(device)

    generated_ids = student_model.generate(model_inputs.input_ids, max_new_tokens=512, do_sample=True)

    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                     zip(model_inputs.input_ids, generated_ids)]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(response)

if __name__ == "__main__":
    main()










