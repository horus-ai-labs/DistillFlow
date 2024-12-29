from typing import Optional, Union
import torch.nn as nn
from datasets import IterableDataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from trl import SFTTrainer, SFTConfig
import torch
import torch.nn.functional as F

from ..common import get_current_device
from ..distill_datasets.loader import DatasetModule


# TODO: Change default value behaviour. Throw warning to users for default values.
class LogitsTrainer(SFTTrainer):
    def __init__(self,
                 model: Union[PreTrainedModel, nn.Module, str],
                 dataset_module: DatasetModule,
                 args: Optional[SFTConfig] = None,
                 tokenizer: Optional[PreTrainedTokenizerBase] = None,
                 max_seq_length: Optional[int] = None,
                 dataset_text_field: Optional[str] = None,
                 teacher_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
                 distillation_args: Optional[dict] = None,
                 ):
        self.teacher_model = teacher_model
        self.distillation_args = distillation_args
        train_dataset = dataset_module["train_dataset"]
        eval_dataset = dataset_module["eval_dataset"]
        self.max_seq_length = max_seq_length
        self.device = get_current_device()

        if isinstance(train_dataset, IterableDataset) and args.max_steps == -1:
            raise ValueError("max steps should be specified when using dataset with streaming mode enabled.")

        super().__init__(model=model, args=args, train_dataset=train_dataset,
                         eval_dataset=eval_dataset, tokenizer=tokenizer,
                         max_seq_length=max_seq_length,
                         dataset_text_field=dataset_text_field)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        # inputs.set_format("torch")
        # self.print_input(inputs)
        self.teacher_model = self.teacher_model.to(inputs['labels'].device) if self.device.type == "mps" else self.teacher_model

        student_model = model.module if hasattr(model, 'module') else model
        teacher_model = self.teacher_model.module if hasattr(self.teacher_model, 'module') else self.teacher_model

        student_outputs = student_model(**inputs)
        with torch.no_grad():
            teacher_outputs = teacher_model(**inputs)

        # self.print_output(student_outputs)
        # self.print_output(teacher_outputs)

        custom_loss = self.distillation_loss(student_outputs.logits, teacher_outputs.logits,
                                             inputs, student_outputs.loss)
        return (custom_loss, student_outputs) if return_outputs else custom_loss

    def distillation_loss(self, student_logits, teacher_logits, inputs, original_loss):
        student_logits, teacher_logits = student_logits.to(inputs['labels'].device), teacher_logits.to(inputs['labels'].device)

        student_logits_scaled = student_logits / self.distillation_args["temperature"]
        teacher_logits_scaled = teacher_logits / self.distillation_args["temperature"]

        loss_kd = F.kl_div(
            F.log_softmax(student_logits_scaled, dim=-1),
            F.softmax(teacher_logits_scaled, dim=-1),
            reduction='batchmean'
        ) * (self.distillation_args["temperature"] ** 2) / self.max_seq_length

        return self.distillation_args["alpha"] * loss_kd + (1 - self.distillation_args["alpha"]) * original_loss


    def print_input(self, inputs):
        input_texts = self.tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True)
        print(input_texts)

    def print_output(self, outputs):
        output_text = self.tokenizer.batch_decode(torch.argmax(outputs.logits, dim=-1), skip_special_tokens=True)
        print(output_text)