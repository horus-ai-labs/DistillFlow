import torch

from datasets import Dataset
from distillflow.model.args import ModelArgs
from distillflow.model.finetuning_args import FinetuningArguments
from distillflow.model.loader import load_model, load_tokenizer
from distillflow.trainer.logits_distillation import LogitsTrainer

# Dummy dataset
data = {
    "input_ids": [[101, 2001, 2023, 102], [101, 2043, 2003, 102]],
    "attention_mask": [[1, 1, 1, 1], [1, 1, 1, 1]],
    "labels": [[2001, 2023, 2003, 102], [2043, 2003, 101, 102]],
}
dataset = Dataset.from_dict(data)

# Teacher and student models
model_config = ModelArgs(
    model_name_or_path="gpt2"
)
teacher = load_model(model_config, FinetuningArguments(), False)
student = load_model(model_config, FinetuningArguments(), True)

# Tokenizer
tokenizer = load_tokenizer(model_config)

def test_basic_distillation(inputs, distillation_args, expected_loss):
    # Initialize the trainer
    trainer = LogitsTrainer(
        accelerator=None,
        model=student,
        dataset_module={"train_dataset": dataset, "eval_dataset": dataset},
        args=None,
        tokenizer=tokenizer,
        max_seq_length=512,
        teacher_model=teacher,
        distillation_args=distillation_args,
    )

    inputs = {k: v.to(student.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

    # Compute loss
    loss = trainer.compute_loss(student, inputs)

    # Verify the loss is a scalar tensor
    assert isinstance(loss, torch.Tensor), "Loss should be a torch.Tensor."
    assert loss.ndim == 0, "Loss should be a scalar tensor."
    assert loss.item() == expected_loss, f"Incorrect loss value. Expected {expected_loss} was {loss.item()}"

def test_logits_padding():
    trainer = LogitsTrainer(
        accelerator=None,
        model=student,
        dataset_module={"train_dataset": dataset, "eval_dataset": dataset},
        args=None,
        tokenizer=tokenizer,
        max_seq_length=512,
        teacher_model=teacher,
        distillation_args=None,
    )

    student_logits = torch.randn(2, 4)
    teacher_logits = torch.randn(2, 6)

    padded_student_logits, padded_teacher_logits = trainer.pad_logits(student_logits, teacher_logits)

    # Validate dimensions
    assert padded_student_logits.shape == (2, 6), "Student logits should be padded to match teacher logits."
    assert padded_teacher_logits.shape == (2, 6), "Teacher logits should remain unchanged."

def test_temperature_scaling(distillation_args, expected_loss):
    from transformers import AutoModelForCausalLM

    trainer = LogitsTrainer(
        accelerator=None,
        model=student,
        dataset_module={"train_dataset": dataset, "eval_dataset": dataset},
        args=None,
        tokenizer=tokenizer,
        max_seq_length=512,
        teacher_model=teacher,
        distillation_args=distillation_args,
    )

    student_logits = torch.randn(2, 5)
    teacher_logits = torch.randn(2, 5)
    inputs = {"labels": torch.tensor([0, 1])}

    original_loss = torch.tensor(0.5)

    loss = trainer.distillation_loss(student_logits, teacher_logits, inputs, original_loss)

    # Ensure the loss is computed
    assert isinstance(loss, torch.Tensor), "Loss should be a torch.Tensor."
    assert loss.ndim == 0, "Loss should be a scalar tensor."
    assert loss.item() == expected_loss, f"Incorrect loss value. Expected {expected_loss} was {loss.item()}"

# Dummy inputs
inputs1 = {
    "input_ids": torch.tensor([[101, 2001, 2023, 102]]),
    "attention_mask": torch.tensor([[1, 1, 1, 1]]),
    "labels": torch.tensor([[2001, 2023, 2003, 102]]),
}
inputs2 = {
    "input_ids": torch.tensor([[0, 0, 0, 0]]),
    "attention_mask": torch.tensor([[1, 1, 1, 1]]),
    "labels": torch.tensor([[2001, 2023, 2003, 102]]),
}
inputs3 = {
    "input_ids": torch.tensor([[0, 0, 0, 0]]),
    "attention_mask": torch.tensor([[1, 1, 1, 1]]),
    "labels": torch.tensor([[0, 0, 0, 0]]),
}
inputs4 = {
    "input_ids": torch.tensor([[0, 0, 0, 0]]),
    "attention_mask": torch.tensor([[0, 0, 0, 0]]),
    "labels": torch.tensor([[0, 0, 0, 0]]),
}

test_cases = [
    {"func": test_basic_distillation, "args": [inputs1, {"temperature": 2.0, "alpha": 0.7,}, 4.338196754455566]},
    {"func": test_basic_distillation, "args": [inputs1, {"temperature": 3.0, "alpha": 0.7,}, 4.338056564331055]},
    {"func": test_basic_distillation, "args": [inputs2, {"temperature": 3.0, "alpha": 0.7,}, 4.720443248748779]},
    {"func": test_basic_distillation, "args": [inputs3, {"temperature": 3.0, "alpha": 0.7,}, 1.385455846786499]},
    {"func": test_basic_distillation, "args": [inputs4, {"temperature": 3.0, "alpha": 0.7,}, 0.27520856261253357]},
    {"func": test_logits_padding, "args": []},
    {"func": test_temperature_scaling, "args": [{"temperature": 2.0, "alpha": 1.7}, -0.34839552640914917]}
]

for i, test in enumerate(test_cases, 1):
    test["func"](*test["args"])  # Unpack and pass the arguments
    print(f"Test Case {i} passed")
