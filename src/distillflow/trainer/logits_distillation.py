from typing import Callable, Dict, List, Optional, Tuple, Union
import torch.nn as nn
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from trl import SFTTrainer, SFTConfig

# TODO: Change default value behaviour. Throw warning to users for default values.
class LogitsTrainer(SFTTrainer):
    def __init__(self,
                 model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
                 args: Optional[SFTConfig] = None,
                 train_dataset: Optional[Dataset] = None,
                 eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
                 tokenizer: Optional[PreTrainedTokenizerBase] = None,
                 max_seq_length: Optional[int] = None,
                 dataset_text_field: Optional[str] = None,
                 teacher_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
                 distillation_args: Optional[dict] = None,
                 tokenizer_args: Optional[dict] = None,
                 ):
        self.teacher_model = teacher_model
        self.distillation_args = distillation_args
        self.tokenizer_args = tokenizer_args

        super().__init__(model=model, args=args, train_dataset=train_dataset,
                         eval_dataset=eval_dataset, tokenizer=tokenizer,
                         max_seq_length=max_seq_length,
                         dataset_text_field=dataset_text_field)

    def compute_loss(self, model, inputs, return_outputs=False):
        inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        self.teacher_model = self.teacher_model.to(model.device)

        student_model = model.module if hasattr(model, 'module') else model
        teacher_model = self.teacher_model.module if hasattr(self.teacher_model, 'module') else self.teacher_model

        student_outputs = student_model(**inputs)
        with torch.no_grad():
            teacher_outputs = teacher_model(**inputs)

        custom_loss = self.distillation_loss(student_outputs.logits, teacher_outputs.logits, inputs,
                                             student_outputs.loss)
        return (custom_loss, student_outputs) if return_outputs else custom_loss

    def distillation_loss(self, student_logits, teacher_logits, inputs, original_loss):
        student_logits, teacher_logits = pad_logits(student_logits.to(self.model.device),
                                                    teacher_logits.to(self.model.device))

        student_logits_scaled = student_logits / self.distillation_args["temperature"]
        teacher_logits_scaled = teacher_logits / self.distillation_args["temperature"]

        loss_kd = F.kl_div(
            F.log_softmax(student_logits_scaled, dim=-1),
            F.softmax(teacher_logits_scaled, dim=-1),
            reduction='batchmean'
        ) * (self.distillation_args["temperature"] ** 2) / self.tokenizer_args["max_length"]

        return self.distillation_args["alpha"] * loss_kd + (1 - self.distillation_args["alpha"]) * original_loss



