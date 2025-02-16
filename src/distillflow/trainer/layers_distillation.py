from accelerate import Accelerator
from datasets import IterableDataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase
import torch
from torch.nn import functional as F
from trl import SFTTrainer

from .AdaptationLayer import AdaptationLayer
from .args import DistillArgs
from ..common import get_current_device
from ..datasets.loader import DatasetModule

class LayersTrainer(SFTTrainer):
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

        custom_loss = self.distillation_loss(student_outputs, teacher_outputs, inputs, original_loss)  / self.args.gradient_accumulation_steps
        return (custom_loss, student_outputs) if return_outputs else custom_loss

    def distillation_loss(self, student_outputs, teacher_outputs, inputs, original_loss):
        student_hidden_states = student_outputs.hidden_states
        teacher_hidden_states = teacher_outputs.hidden_states

        self.adaptation_layer = self.adaptation_layer.to(self.device)
        adapted_student_hidden_states = self.adaptation_layer(student_hidden_states)

        total_loss_kd = 0
        for student_hidden, teacher_idx in self.adaptation_layer.layer_mapping.items():
            if self.distill_args.strategy == "weighted":
                teacher_hidden = torch.zeros_like(teacher_hidden_states[0])
                for idx, weight in teacher_idx.items():
                    teacher_hidden = weight * teacher_hidden_states[idx]
            else:
                teacher_hidden = teacher_hidden_states[teacher_idx]

            if adapted_student_hidden_states[student_hidden].shape != teacher_hidden.shape:
                raise ValueError(
                    f"Shape mismatch: student {adapted_student_hidden_states[student_hidden].shape} vs teacher {teacher_hidden.shape}")

            student_probs = F.softmax(
                adapted_student_hidden_states[student_hidden] / self.distill_args.temperature, dim=-1)
            teacher_probs = F.softmax(teacher_hidden / self.distill_args.temperature, dim=-1)

            loss_kd = F.kl_div(
                F.log_softmax(adapted_student_hidden_states[student_hidden] / self.distill_args.temperature,
                              dim=-1),
                teacher_probs,
                reduction='batchmean'
            ) * (self.distill_args.temperature ** 2)

            total_loss_kd += loss_kd

        avg_loss_kd = total_loss_kd / len(self.adaptation_layer.layer_mapping)
        hidden_dim = adapted_student_hidden_states[0].size(-1)
        scaled_loss_kd = avg_loss_kd / hidden_dim

        total_loss = self.distill_args.alpha * scaled_loss_kd + (
                    1 - self.distill_args.alpha) * original_loss
        return total_loss