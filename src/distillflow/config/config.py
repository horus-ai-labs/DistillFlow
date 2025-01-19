from typing import Optional

from pydantic import BaseModel, Field

from distillflow.datasets.args import DataArgs
from distillflow.model.args import ModelArgs, TokenizerArgs
from distillflow.trainer.args import DistillArgs

class Config(BaseModel):
    student_model: ModelArgs = Field(
        description="Details about the student model that we want to train"
    )
    teacher_model: ModelArgs = Field(
        description="Details about the teacher model"
    )
    data: DataArgs = Field(
        description="The datasets that we want to choose to run the training"
    )
    tokenizer: Optional[TokenizerArgs] = Field(
        default=None,
        description="Tokenizer specific arguments"
    )
    distill: DistillArgs = Field(
        description="Distillation training parameters"
    )

    model_config = {
        "extra": "forbid"
    }