from typing import Dict, Any
import json
import yaml
import argparse
import re

from tqdm import tqdm
from accelerate import Accelerator
from pydantic import ValidationError
from datasets import load_dataset

import s3_utils
from torch.utils.data import DataLoader
from distillflow.config import Config
from distillflow.common import get_current_device
from distillflow.config.validator import print_validation_error
from distillflow.datasets.args import DataArgs, DatasetArgs
from distillflow.datasets.loader import get_dataset
from distillflow.datasets.template.args import TemplateArgs
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

def append_to_jsonl(data, filename: str) -> None:
    """Append a json payload to the end of a jsonl file."""
    json_string = json.dumps(data)
    with open(filename, "a") as f:
        f.write(json_string + "\n")

def prepare_model(model_config, accelerator, accelerator_state, finetuning_args, is_trainable):
    """Prepare model for training."""

    model = load_model(model_config, finetuning_args=finetuning_args,
                               is_trainable=is_trainable)
    if accelerator is not None:
        accelerator.state.select_deepspeed_plugin(accelerator_state)
        model = accelerator.prepare(model)

    return model

def extract_number(text):
    pattern = r"(?i)the correct answer is\s+(\d+):"

    match = re.search(pattern, text)
    if match:
        extracted_number = match.group(1)
        # print(f"Extracted number: {extracted_number}")
        return int(extracted_number)
    else:
        return None

def acc(pred, gt):
    if pred == gt:
        return 1.0
    else:
        return 0.0

def main():
    args = parse_args()
    try:
        config = Config(**(load_config(args.config)))

    except ValidationError as error:
        print_validation_error(error)
        return


    device = get_current_device()

    # Load Single Model (by default student is loaded and teacher is ignored).
    accelerator = None
    student_plugin = DeepSpeedPlugin(hf_ds_config=config.student_model.deepspeed_config)
    deepspeed_plugins = {"student": student_plugin}

    if device.type != "mps":
        accelerator = Accelerator(deepspeed_plugins=deepspeed_plugins)

    student_model = prepare_model(config.student_model, accelerator=accelerator, accelerator_state='student',
                                  finetuning_args=FinetuningArguments(), is_trainable=False)

    # Load tokenizer and dataset
    tokenizer_template = config.tokenizer.template
    # Auto-regressive inference. padding on the left.
    tokenizer = load_tokenizer(config.student_model, template=tokenizer_template, padding_side="left")
    tokenizer.eos_token = "<|im_end|>"

    def tokenizer_function(examples):
        return tokenizer(examples[config.data.text_field], truncation=True, max_length=config.distill.max_seq_length,
                         padding="max_length", return_tensors="pt")

    # dataset = get_dataset(data_args=config.data, tokenizer=tokenizer, tokenizer_function=None)
    #
    # # def fetch(examples: Dict[str, Any]) -> Dict[str, Any]:
    # #     question = None
    # #     for prompt in examples["_prompt"]:
    # #         if prompt["role"] == "user":
    # #             question = prompt
    # #
    # #     response = None
    # #     for prompt in examples["_response"]:
    # #         if prompt["role"] == "assistant":
    # #             response = prompt["content"]
    # #
    # #     if question is None or response is None:
    # #         return {}
    # #
    # #     return {"question": question, "response": response}
    #
    # # dataset = dataset["train_dataset"].map(fetch, batched=False, remove_columns=dataset["train_dataset"].column_names)
    # print(dataset['train_dataset'][0])

    dataset =  load_dataset(
            path=config.data.train_datasets[0].path,
            split=config.data.train_datasets[0].split,
            cache_dir=config.data.cache_dir,
            token=config.data.hf_hub_token,
            streaming=config.data.streaming, # and (dataset_attr.load_from != "file")),
            trust_remote_code=True
        )

    def sharegpt_format(example):
        conversations = example['conversations']
        message = []
        answer = []

        if isinstance(conversations, list):
            for conversation in conversations:
                if isinstance(conversation, dict):
                    if conversation.get('from') == 'human':
                        message.append({"role": "user", "content": conversation.get('value', '')})
                    elif conversation.get('from') == 'gpt':
                        answer.append({"role": "assistant", "content": conversation.get('value', '')})
                    elif conversation.get('from') == 'system':
                        message.insert(0, {"role": "system", "content": conversation.get('value', '')})

        if not any(msg.get('role') == 'system' for msg in message):
            message.insert(0, {"role": "system", "content": "You are a helpful assistant."})

        text = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

        return {"text": text}, answer

    # Preprocess and tokenize the dataset
    print("Preprocessing and tokenizing dataset...")
    original_columns = dataset.column_names
    dataset = dataset.map(sharegpt_format, remove_columns=original_columns)

    if device.type == "mps":
        student_model = student_model.to(device)

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,  # Shuffle the dataset for each epoch
        num_workers=8,  # Use multiprocessing
        pin_memory=True,  # Speeds up data transfer to GPU
        prefetch_factor=2  # Number of batches prefetched by each worker
    )

    for data in tqdm(dataloader):
        print(data)
        model_inputs, answer = sharegpt_format(data)
        print(model_inputs)
        model_inputs = tokenizer(model_inputs[config.data.text_field], truncation=True,
                  padding=True, return_tensors="pt").to(device)

        print(model_inputs['attention_mask'].shape)
        # print(processed[0], processed[1])
        # model_inputs = model_inputs.to(device)


        generated_ids = student_model.generate(input_ids = model_inputs.input_ids,
                                               attention_mask = model_inputs.attention_mask,
                                               max_new_tokens=25, do_sample=False)

        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                         zip(model_inputs.input_ids, generated_ids)]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print("Response", extract_number(response))
        print("Answer", extract_number(answer[0]['content']))

        append_to_jsonl({'resonse': response, 'answer': answer[0]['content']},
                        './results/infer_finetuned.jsonl')
        metric = acc(extract_number(response), extract_number(answer[0]['content']))
        print(metric)


if __name__ == "__main__":
    main()










