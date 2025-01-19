import unittest

import torch

from datasets import Dataset
from trl import SFTConfig

from distillflow.model.args import ModelArgs
from distillflow.model.finetuning_args import FinetuningArguments
from distillflow.model.loader import load_model, load_tokenizer
from distillflow.trainer.args import DistillArgs
from distillflow.trainer.logits_distillation import LogitsTrainer

class TestDataset(unittest.TestCase):

    def setUp(self):
        # Dummy dataset
        data = {
            "input_ids": [[101, 2001, 2023, 102], [101, 2043, 2003, 102]],
            "attention_mask": [[1, 1, 1, 1], [1, 1, 1, 1]],
            "labels": [[2001, 2023, 2003, 102], [2043, 2003, 101, 102]],
        }
        self.dataset = Dataset.from_dict(data)

        # Teacher and student models
        model_config = ModelArgs(
            model_name_or_path="gpt2"
        )
        self.teacher = load_model(model_config, FinetuningArguments(), False)
        self.student = load_model(model_config, FinetuningArguments(), True)

        # Tokenizer
        self.tokenizer = load_tokenizer(model_config)
        self.sft_config = SFTConfig(
            output_dir="./results",
            num_train_epochs= 1,
            per_device_train_batch_size= 1,
            gradient_accumulation_steps= 1
        )

    def _test_basic_distillation(self, inputs, extra_args, expected_loss):
        # Initialize the trainer
        args = DistillArgs(
            max_seq_length=512,
            sft_config = self.sft_config,
            **extra_args
        )
        trainer = LogitsTrainer(
            accelerator=None,
            model=self.student,
            dataset_module={"train_dataset": self.dataset, "eval_dataset": self.dataset},
            tokenizer=self.tokenizer,
            distill_args=args,
            teacher_model=self.teacher
        )

        inputs = {k: v.to(self.student.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

        # Compute loss
        loss = trainer.compute_loss(self.student, inputs)

        # Verify the loss is a scalar tensor
        assert isinstance(loss, torch.Tensor), "Loss should be a torch.Tensor."
        assert loss.ndim == 0, "Loss should be a scalar tensor."
        assert loss.item() == expected_loss, f"Incorrect loss value. Expected {expected_loss} was {loss.item()}"

    def test_temperature_alpha(self):
        # Dummy inputs
        inputs1 = {
            "input_ids": torch.tensor([[101, 2001, 2023, 102]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
            "labels": torch.tensor([[2001, 2023, 2003, 102]]),
        }
        self._test_basic_distillation(inputs1, {"temperature": 2.0, "alpha": 0.7,}, 4.338196754455566)
        self._test_basic_distillation(inputs1, {"temperature": 3.0, "alpha": 0.7,}, 4.338056564331055)

    def test_all_input_ids_0s(self):
        # Dummy inputs
        inputs2 = {
            "input_ids": torch.tensor([[0, 0, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
            "labels": torch.tensor([[2001, 2023, 2003, 102]]),
        }
        self._test_basic_distillation(inputs2, {"temperature": 3.0, "alpha": 0.7,}, 4.720443248748779)

    def test_all_input_ids_labels_0s(self):
        # Dummy inputs
        inputs3 = {
            "input_ids": torch.tensor([[0, 0, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
            "labels": torch.tensor([[0, 0, 0, 0]]),
        }
        self._test_basic_distillation(inputs3, {"temperature": 3.0, "alpha": 0.7,}, 1.385455846786499)

    def test_all_inputs_0s(self):
        inputs4 = {
            "input_ids": torch.tensor([[0, 0, 0, 0]]),
            "attention_mask": torch.tensor([[0, 0, 0, 0]]),
            "labels": torch.tensor([[0, 0, 0, 0]]),
        }
        self._test_basic_distillation(inputs4, {"temperature": 3.0, "alpha": 0.7,}, 0.27520856261253357)

    def test_logits_padding(self):
        trainer = LogitsTrainer(
            accelerator=None,
            model=self.student,
            dataset_module={"train_dataset": self.dataset, "eval_dataset": self.dataset},
            distill_args=DistillArgs(
                max_seq_length=512,
                sft_config=self.sft_config
            ),
            tokenizer=self.tokenizer,
            teacher_model=self.teacher,
        )

        student_logits = torch.randn(2, 4)
        teacher_logits = torch.randn(2, 6)

        padded_student_logits, padded_teacher_logits = trainer.pad_logits(student_logits, teacher_logits)

        # Validate dimensions
        assert padded_student_logits.shape == (2, 6), "Student logits should be padded to match teacher logits."
        assert padded_teacher_logits.shape == (2, 6), "Teacher logits should remain unchanged."

    def test_temperature_scaling(self):

        trainer = LogitsTrainer(
            accelerator=None,
            model=self.student,
            dataset_module={"train_dataset": self.dataset, "eval_dataset": self.dataset},
            distill_args=DistillArgs(
                max_seq_length=512,
                sft_config=self.sft_config,
                temperature=2.0,
                alpha=1.7
            ),
            tokenizer=self.tokenizer,
            teacher_model=self.teacher,
        )

        student_logits = torch.randn(2, 5)
        teacher_logits = torch.randn(2, 5)
        inputs = {"labels": torch.tensor([0, 1])}

        original_loss = torch.tensor(0.5)

        loss = trainer.distillation_loss(student_logits, teacher_logits, inputs, original_loss)

        # Ensure the loss is computed
        expected_loss = -0.34839552640914917
        assert isinstance(loss, torch.Tensor), "Loss should be a torch.Tensor."
        assert loss.ndim == 0, "Loss should be a scalar tensor."
        assert loss.item() == expected_loss, f"Incorrect loss value. Expected {expected_loss} was {loss.item()}"

if __name__ == "__main__":
    unittest.main()