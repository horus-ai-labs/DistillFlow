from accelerate import Accelerator
from datasets import IterableDataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from trl import SFTTrainer
import torch
import torch.nn.functional as F

from .args import DistillArgs
from ..common import get_current_device
from ..datasets.loader import DatasetModule

class FineTuning(SFTTrainer):
    def __init__(self,
                 accelerator: Accelerator,
                 distill_args: DistillArgs,
                 teacher_model: PreTrainedModel,
                 model: PreTrainedModel,
                 dataset_module: DatasetModule,
                 tokenizer: PreTrainedTokenizerBase
                 ):
        self.accelerator = accelerator
        self.distill_args = distill_args
        self.teacher_model = teacher_model
        train_dataset = dataset_module["train_dataset"]
        eval_dataset = dataset_module["eval_dataset"]
        self.device = get_current_device()
        if self.device.type == 'mps':
            # Explicitly place the models on device since accelerate prepare does not work on MPS.
            self.teacher_model = self.teacher_model.to(self.device)
            model = model.to(self.device)

        #TODO: remove excess code.
        # if self.accelerator is not None:
        #     self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        #     if self.is_deepspeed_enabled:
        #         if not (
        #                 getattr(teacher_model.pretrained_model, "is_loaded_in_8bit", False)
        #                 or getattr(teacher_model.pretrained_model, "is_loaded_in_4bit", False)
        #         ):  # quantized models are already set on the correct device
        #             self.teacher_model = self._prepare_deepspeed(self.teacher_model)

        if isinstance(train_dataset, IterableDataset) and distill_args.sft_config.max_steps == -1:
            raise ValueError("max steps should be specified when using dataset with streaming mode enabled.")

        super().__init__(model=model, args=distill_args.sft_config, train_dataset=train_dataset,
                         eval_dataset=eval_dataset, tokenizer=tokenizer,
                         max_seq_length=distill_args.max_seq_length,
                         dataset_text_field=distill_args.dataset_text_field)


    def output(self, model, inputs, no_grad):
        if no_grad:
            with torch.no_grad():
                return model(**inputs)
        else:
            return model(**inputs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        student_model = model.module if hasattr(model, 'module') else model
        student_outputs = self.output(student_model, inputs, False)

        return student_outputs.loss
