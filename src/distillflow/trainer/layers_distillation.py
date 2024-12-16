from typing import Union, Optional

from datasets import IterableDataset
from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase
import torch
from torch.nn import functional as F
from trl import SFTConfig, SFTTrainer

from ..common import get_current_device
from ..distill_datasets.loader import DatasetModule

class AdaptationLayer(torch.nn.Module):
    def __init__(self, student_dim, teacher_dim, num_student_layers, num_teacher_layers, strategy="interpolate",
                 dtype=torch.bfloat16, selection_indices=None, weights=None):
        super().__init__()
        self.projections = torch.nn.ModuleList([
            torch.nn.Linear(student_dim, teacher_dim, dtype=dtype)
            for _ in range(num_student_layers)
        ])
        # self.layer_mapping = self.create_layer_mapping(num_student_layers, num_teacher_layers)
        self.layer_mapping = self.map_teacher_to_student_layers(num_student_layers, num_teacher_layers, strategy, selection_indices, weights)
        self.dtype = dtype

    def map_teacher_to_student_layers(self, num_student_layers, num_teacher_layers, strategy="select",
                                      selection_indices=None, weights=None) -> {}:
        """
        Maps teacher model layers to student model layers based on the specified strategy.

        Args:
            num_student_layers (int): Number of layers in the student model.
            num_teacher_layers (int): Number of layers in the teacher model.
            strategy (str): Layer mapping strategy ("direct", "select", "interpolate", "weighted").
            selection_indices (list): Specific teacher layers to select (used when strategy="select").
            weights (list of lists): Weights for combining teacher layers for each student layer
                                     (used when strategy="weighted").

        Returns:
            list: List of mapping indices or weights from teacher layers to align with student layers.
        """
        if strategy == "direct":
            # Direct one-to-one mapping
            return {
                i: i
                for i in range(num_student_layers)
            }

        elif strategy == "select":
            # Use specific teacher layers for mapping
            if selection_indices is None:
                raise ValueError("selection_indices must be provided for 'select' strategy.")
            if len(selection_indices) != num_student_layers:
                raise ValueError("Number of selection_indices must match num_student_layers.")
            return {
                i: teacher_layer
                for i, teacher_layer in enumerate(selection_indices)
            }

        elif strategy == "interpolate":
            # Interpolate teacher layers to match student layers
            return {
                i: round(i * (num_teacher_layers - 1) / (num_student_layers - 1))
                for i in range(num_student_layers)
            }
        elif strategy == "weighted":
            # Weighted combination of teacher layers for each student layer
            if weights is None:
                raise ValueError("weights must be provided for 'weighted' strategy.")
            if len(weights) != num_student_layers:
                raise ValueError("Number of weight sets must match num_student_layers.")
            return {
                i: {j: weight for j, weight in enumerate(weight_set) if weight > 0}
                for i, weight_set in enumerate(weights)
            }
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def forward(self, student_hidden_states):
        adapted_hidden_states = []
        for i, hidden_state in enumerate(student_hidden_states):
            if i >= len(self.projections):
                break
            adapted_hidden_states.append(self.projections[i](hidden_state.to(self.dtype)))
        return adapted_hidden_states

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

    def compute_loss(self, model, inputs, return_outputs=False):
        model_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
        labels = inputs["labels"]
        student_outputs = model(**model_inputs, labels=labels, output_hidden_states=True)
        original_loss = student_outputs.loss

        self.teacher_model = self.teacher_model.to(self.device)

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