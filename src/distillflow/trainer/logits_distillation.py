from typing import Optional, Union
import torch.nn as nn
from accelerate import Accelerator
from datasets import IterableDataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from trl import SFTTrainer, SFTConfig
import torch
import torch.nn.functional as F
import time
from ..common import get_current_device
from ..distill_datasets.loader import DatasetModule


# TODO: Change default value behaviour. Throw warning to users for default values.
class LogitsTrainer(SFTTrainer):
    def __init__(self,
                 accelerator: Accelerator,
                 model: Union[PreTrainedModel, nn.Module, str],
                 dataset_module: DatasetModule,
                 args: Optional[SFTConfig] = None,
                 tokenizer: Optional[PreTrainedTokenizerBase] = None,
                 max_seq_length: Optional[int] = None,
                 dataset_text_field: Optional[str] = None,
                 teacher_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
                 distillation_args: Optional[dict] = None,
                 ):
        self.accelerator = accelerator
        self.teacher_model = teacher_model
        self.distillation_args = distillation_args
        train_dataset = dataset_module["train_dataset"]
        eval_dataset = dataset_module["eval_dataset"]
        self.max_seq_length = max_seq_length
        self.device = get_current_device()

        if self.accelerator is not None:
            self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
            if self.is_deepspeed_enabled:
                if not (
                        getattr(teacher_model.pretrained_model, "is_loaded_in_8bit", False)
                        or getattr(teacher_model.pretrained_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.teacher_model = self._prepare_deepspeed(self.teacher_model)
         #   else:
         #       self.teacher_model = self.accelerator.prepare(self.teacher_model)
         #       model = self.accelerator.prepare(model)

        if isinstance(train_dataset, IterableDataset) and args.max_steps == -1:
            raise ValueError("max steps should be specified when using dataset with streaming mode enabled.")

        super().__init__(model=model, args=args, train_dataset=train_dataset,
                         eval_dataset=eval_dataset, tokenizer=tokenizer,
                         max_seq_length=max_seq_length,
                         dataset_text_field=dataset_text_field)

    def pad_logits(self, student_logits, teacher_logits):
        student_size, teacher_size = student_logits.size(-1), teacher_logits.size(-1)
        if student_size != teacher_size:
            pad_size = abs(student_size - teacher_size)
            pad_tensor = torch.zeros((*teacher_logits.shape[:-1], pad_size), dtype=teacher_logits.dtype,
                                     device=teacher_logits.device)
            return (
            torch.cat([student_logits, pad_tensor], dim=-1), teacher_logits) if student_size < teacher_size else (
            student_logits, torch.cat([teacher_logits, pad_tensor], dim=-1))
        return student_logits, teacher_logits

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        # inputs.set_format("torch")
        start = time.time()
        # self.print_input(inputs)
        # self.teacher_model = self.teacher_model.to(inputs['labels'].device) if self.device.type == "mps" else self.teacher_model
        inputs = self.accelerator.prepare(inputs)
        student_model = model.module if hasattr(model, 'module') else model
        teacher_model = self.teacher_model.module if hasattr(self.teacher_model, 'module') else self.teacher_model
        endprep = time.time()
        print("Prep time:")
        print(endprep-start)

        start = time.time()
        student_outputs = student_model(**inputs)
        end1 = time.time()
        print("Student time:")
        print(end1-start)
        start1 = time.time()
        with torch.no_grad():
            teacher_outputs = teacher_model(**inputs)
            end2 = time.time()
            print("Teacher time:")
            print(end2-start)
            print(end2-start1)

            # print(teacher_outputs.keys())
            #
            # teacher_outputs2 = teacher_model.generate(**inputs, max_new_tokens=200,
            #                                             temperature=0.7,
            #                                             repetition_penalty=1.2)

        # print("Printing the Student Output: ")
        # self.print_output(student_outputs)
        # print("Printing the Teacher Output: ")
        # self.print_output(teacher_outputs)
        # print("Printing the Teacher Output 2")
        # print(self.tokenizer.batch_decode(teacher_outputs2))
        # exit()
        start = time.time()
        custom_loss = self.distillation_loss(student_outputs.logits, teacher_outputs.logits,
                                             inputs, student_outputs.loss)

        endloss = time.time()
        print("loss time:")
        print(endloss-start)
        return (custom_loss, student_outputs) if return_outputs else custom_loss

    def distillation_loss(self, student_logits, teacher_logits, inputs, original_loss):
        student_logits, teacher_logits = student_logits.to(inputs['labels'].device), teacher_logits.to(inputs['labels'].device)
        student_logits, teacher_logits = self.pad_logits(student_logits, teacher_logits)

        student_logits_scaled = student_logits / self.distillation_args["temperature"]
        teacher_logits_scaled = teacher_logits / self.distillation_args["temperature"]

        loss_kd = F.kl_div(
            F.log_softmax(student_logits_scaled, dim=-1),
            F.softmax(teacher_logits_scaled, dim=-1),
            reduction='batchmean'
        ) * (self.distillation_args["temperature"] ** 2) / self.max_seq_length

        return self.distillation_args["alpha"] * loss_kd + (1 - self.distillation_args["alpha"]) * original_loss


    def print_input(self, inputs):
        print("INPUT:::::")
        input_texts = self.tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=False)
        print(input_texts)

    def print_output(self, outputs):
        print("OUTPUT:::::")
        output_text = self.tokenizer.batch_decode(torch.argmax(outputs.logits, dim=-1), skip_special_tokens=False)
        print(output_text)

    def print_generate_output(self, input_ids):
        outputs = self.teacher_model.generate(
            input_ids=input_ids,
            max_length=512,
            num_beams=5,
            early_stopping=True
        )
        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
        print(decoded_outputs)
