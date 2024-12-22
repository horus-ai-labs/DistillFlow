from typing import Union, Optional

from datasets import IterableDataset
from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase
import torch
from torch.nn import functional as F
from trl import SFTConfig, SFTTrainer

from .AdaptationLayer import AdaptationLayer
from ..common import get_current_device
from ..distill_datasets.loader import DatasetModule

class LayersTrainer(SFTTrainer):
    def __init__(self,
                 model: Union[PreTrainedModel, nn.Module, str],
                 dataset_module: DatasetModule,
                 args: Optional[SFTConfig] = None,
                 tokenizer: Optional[PreTrainedTokenizerBase] = None,
                 max_seq_length: Optional[int] = None,
                 dataset_text_field: Optional[str] = None,
                 teacher_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
                 distillation_args: Optional[dict] = None,
                 tokenizer_args: Optional[dict] = None,
                 strategy="interpolate",
                 selection_indices=None,
                 weights=None
                 ):
        self.teacher_model = teacher_model
        self.distillation_args = distillation_args
        self.tokenizer_args = tokenizer_args
        train_dataset = dataset_module["train_dataset"]
        eval_dataset = dataset_module["eval_dataset"]
        self.device = get_current_device()
        self.strategy = strategy

        self.adaptation_layer = AdaptationLayer(
            model.config.hidden_size,
            teacher_model.config.hidden_size,
            model.config.num_hidden_layers,
            teacher_model.config.num_hidden_layers,
            dtype=torch.float16,
            strategy=strategy,
            selection_indices=selection_indices,
            weights=weights
        ).to(self.device)

        if isinstance(train_dataset, IterableDataset) and args.max_steps == -1:
            raise ValueError("max steps should be specified when using dataset with streaming mode enabled.")

        super().__init__(model=model, args=args, train_dataset=train_dataset,
                         eval_dataset=eval_dataset, tokenizer=tokenizer,
                         max_seq_length=max_seq_length,
                         dataset_text_field=dataset_text_field)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        model_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
        labels = inputs["labels"]
        student_outputs = model(**model_inputs, labels=labels, output_hidden_states=True)
        original_loss = student_outputs.loss

        self.teacher_model = self.teacher_model.to(inputs['labels'].device) if self.device.type == "mps" else self.teacher_model

        teacher_model = self.teacher_model.module if hasattr(self.teacher_model, 'module') else self.teacher_model

        with torch.no_grad():
            teacher_outputs = teacher_model(**model_inputs, output_hidden_states=True)

        custom_loss = self.distillation_loss(student_outputs, teacher_outputs, inputs, original_loss)
        return (custom_loss, student_outputs) if return_outputs else custom_loss

    def distillation_loss(self, student_outputs, teacher_outputs, inputs, original_loss):
        student_hidden_states = student_outputs.hidden_states
        teacher_hidden_states = teacher_outputs.hidden_states

        self.adaptation_layer = self.adaptation_layer.to(self.device)
        adapted_student_hidden_states = self.adaptation_layer(student_hidden_states)

        total_loss_kd = 0
        for student_hidden, teacher_idx in self.adaptation_layer.layer_mapping.items():
            if self.strategy == "weighted":
                teacher_hidden = torch.zeros_like(teacher_hidden_states[0])
                for idx, weight in teacher_idx.items():
                    teacher_hidden = weight * teacher_hidden_states[idx]
            else:
                teacher_hidden = teacher_hidden_states[teacher_idx]

            if adapted_student_hidden_states[student_hidden].shape != teacher_hidden.shape:
                raise ValueError(
                    f"Shape mismatch: student {adapted_student_hidden_states[student_hidden].shape} vs teacher {teacher_hidden.shape}")

            student_probs = F.softmax(
                adapted_student_hidden_states[student_hidden] / self.distillation_args["temperature"], dim=-1)
            teacher_probs = F.softmax(teacher_hidden / self.distillation_args["temperature"], dim=-1)

            loss_kd = F.kl_div(
                F.log_softmax(adapted_student_hidden_states[student_hidden] / self.distillation_args["temperature"],
                              dim=-1),
                teacher_probs,
                reduction='batchmean'
            ) * (self.distillation_args["temperature"] ** 2)

            total_loss_kd += loss_kd

        avg_loss_kd = total_loss_kd / len(self.adaptation_layer.layer_mapping)
        hidden_dim = adapted_student_hidden_states[0].size(-1)
        scaled_loss_kd = avg_loss_kd / hidden_dim

        total_loss = self.distillation_args["alpha"] * scaled_loss_kd + (
                    1 - self.distillation_args["alpha"]) * original_loss
        return total_loss