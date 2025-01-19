from typing import Union, Optional

import torch
from accelerate import Accelerator
from datasets import IterableDataset
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from trl import SFTTrainer

from distillflow.common import get_current_device
from distillflow.datasets.loader import DatasetModule
from distillflow.trainer.AdaptationLayer import AdaptationLayer
from distillflow.trainer.args import DistillArgs

class AttentionTrainer(SFTTrainer):
    def __init__(self,
                 accelerator: Accelerator,
                 distill_args: DistillArgs,
                 teacher_model: PreTrainedModel,
                 model: PreTrainedModel,
                 dataset_module: DatasetModule,
                 tokenizer: PreTrainedTokenizerBase
                 ):
        self.teacher_model = teacher_model
        self.distill_args = distill_args
        train_dataset = dataset_module["train_dataset"]
        eval_dataset = dataset_module["eval_dataset"]
        self.device = get_current_device()
        self.adaptation_layer = AdaptationLayer(
            model.config.hidden_size,
            teacher_model.config.hidden_size,
            model.config.num_hidden_layers,
            teacher_model.config.num_hidden_layers,
            dtype=torch.float16,
            strategy=distill_args.strategy,
            selection_indices=distill_args.selection_indices,
            weights=distill_args.weights
        ).to(self.device)

        if isinstance(train_dataset, IterableDataset) and distill_args.sft_config.max_steps == -1:
            raise ValueError("max steps should be specified when using dataset with streaming mode enabled.")

        super().__init__(model=model, args=distill_args.sft_config, train_dataset=train_dataset,
                         eval_dataset=eval_dataset, tokenizer=tokenizer,
                         max_seq_length=distill_args.max_seq_length,
                         dataset_text_field=distill_args.dataset_text_field)

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
        total_loss = ((1 - self.distill_args.alpha) * loss + self.distill_args.alpha * attention_loss) / self.args.gradient_accumulation_steps

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

        for student_idx, teacher_idx in self.adaptation_layer.layer_mapping.items():
            if self.distill_args.strategy == "weighted":
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
