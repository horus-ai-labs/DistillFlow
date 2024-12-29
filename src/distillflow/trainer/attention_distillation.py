from typing import Union, Optional

import torch
from datasets import IterableDataset
from torch import nn
import torch.nn.functional as F
from transformers import Seq2SeqTrainer, PreTrainedModel, PreTrainedTokenizerBase
from trl import SFTConfig, SFTTrainer

from distillflow.common import get_current_device
from distillflow.distill_datasets.loader import DatasetModule
from distillflow.trainer.AdaptationLayer import AdaptationLayer


class AttentionTrainer(SFTTrainer):
    def __init__(self,
                 model: Union[PreTrainedModel, nn.Module, str],
                 dataset_module: DatasetModule,
                 args: Optional[SFTConfig] = None,
                 tokenizer: Optional[PreTrainedTokenizerBase] = None,
                 max_seq_length: Optional[int] = None,
                 dataset_text_field: Optional[str] = None,
                 teacher_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
                 distillation_args: Optional[dict] = None,
                 strategy="interpolate",
                 selection_indices=None,
                 weights=None
                 ):
        self.teacher_model = teacher_model
        self.distillation_args = distillation_args
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
        # Forward pass for the student model
        student_outputs = model(**inputs, output_attentions=True)

        self.teacher_model = self.teacher_model.to(self.device) if self.device.type == "mps" else self.teacher_model

        teacher_model = self.teacher_model.module if hasattr(self.teacher_model, 'module') else self.teacher_model

        # Forward pass for the teacher model
        with torch.no_grad():
            teacher_outputs = teacher_model(**inputs, output_attentions=True)

        # Primary task loss (e.g., cross-entropy loss)
        loss = student_outputs.loss

        # Attention-based distillation loss
        attention_loss = self.compute_attention_loss(
            teacher_outputs.attentions,
            student_outputs.attentions
        )

        # Combine losses
        total_loss = (1 - self.distillation_args["alpha"]) * loss + self.distillation_args["alpha"] * attention_loss

        return (total_loss, student_outputs) if return_outputs else total_loss

    def compute_attention_loss(self, teacher_attentions, student_attentions):
        """
        Compute attention-based distillation loss.

        Args:
            teacher_attentions: List of teacher model attention maps.
            student_attentions: List of student model attention maps.

        Returns:
            Total attention distillation loss.
        """
        loss = 0.0
        num_layers = len(student_attentions)

        self.adaptation_layer = self.adaptation_layer.to(self.device)
        # adapted_student_hidden_states = self.adaptation_layer(student_hidden_states)

        for student_idx, teacher_idx in self.adaptation_layer.layer_mapping.items():
            if self.strategy == "weighted":
                teacher_attention = torch.zeros_like(teacher_attentions[0])
                for idx, weight in teacher_idx.items():
                    teacher_attention = weight * teacher_attentions[idx]
            else:
                teacher_attention = teacher_attentions[teacher_idx]

            student_attention = student_attentions[student_idx]
            # Align dimensions if needed
            if teacher_attention.size() != student_attention.size():
                teacher_attention = teacher_attention.mean(dim=1, keepdim=True).expand(-1, student_attention.size(1), -1, -1)
                teacher_attention = F.interpolate(
                    teacher_attention,
                    size=student_attention.size()[-2:],  # Resize spatial dimensions
                    mode="bilinear",
                    align_corners=False,
                )

            # MSE Loss for attention maps
            loss += F.mse_loss(student_attention, teacher_attention)

        return loss / num_layers
