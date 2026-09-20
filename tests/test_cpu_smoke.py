"""CPU smoke test: green signal for the modernization migration.

Trains SmolLM2-135M against itself (teacher == student weights) for 2 steps
on CPU and asserts that the logged training loss decreases. Uses HF loaders
directly so the test isolates the trainer plumbing from the model-loader pipeline.
"""

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig

from distillflow.trainer.args import DistillArgs
from distillflow.trainer.logits_distillation import LogitsTrainer


MODEL_ID = "HuggingFaceTB/SmolLM2-135M"


class _LossHistory(TrainerCallback):
    def __init__(self):
        self.losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.losses.append(logs["loss"])


def test_cpu_smoke_loss_decreases(tmp_path):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    teacher = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    student = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    teacher.eval()

    text = "The quick brown fox jumps over the lazy dog."
    dataset = Dataset.from_dict({"text": [text] * 4})

    sft_config = SFTConfig(
        output_dir=str(tmp_path),
        max_steps=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        logging_steps=1,
        report_to=[],
        use_cpu=True,
        save_strategy="no",
    )
    distill_args = DistillArgs(
        max_seq_length=64,
        sft_config=sft_config,
        temperature=2.0,
        alpha=0.1,
    )

    history = _LossHistory()
    trainer = LogitsTrainer(
        accelerator=None,
        model=student,
        dataset_module={"train_dataset": dataset, "eval_dataset": dataset},
        tokenizer=tokenizer,
        distill_args=distill_args,
        teacher_model=teacher,
    )
    trainer.add_callback(history)
    trainer.train()

    assert len(history.losses) >= 2, f"Expected 2+ logged losses, got {history.losses}"
    assert history.losses[-1] < history.losses[0], (
        f"Loss did not decrease across 2 steps: {history.losses}"
    )
