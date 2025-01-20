from typing import Literal, Optional, List, Dict

from pydantic import BaseModel, Field
from trl import SFTConfig

class DistillArgs(BaseModel):
    sft_config: SFTConfig = Field(
        description="SFT Config, with hyperparameters required for training"
    )
    type: Literal["logits", "layers", "attention"] = Field(
        default="logits",
        description="Type of distillation to perform on the given student model with the given dataset (default: logits)"
    )
    max_seq_length: Optional[int] = Field (
        default=4096,
        description="Maximum sequence length to use during training (default: 4096)",
        examples=[1024, 2048, 4096]
    )
    dataset_text_field: Optional[str] = Field(
        default="text",
        description="The key to which the data is mapped",
    )
    temperature: Optional[float] = Field(
        default=0.5,
        description="Temperature"
    )
    alpha: Optional[float] = Field(
        default=2.0,
        description="alpha"
    )
    resume_from_checkpoint: Optional[str] = Field(
        default=None,
        description="Training checkpoint folder path to resume the training from which to resume the training"
    )
    strategy : Literal["direct", "select", "interpolate", "weighted"] = Field(
        default = "interpolate",
        description="Strategy to select when mapping the teacher and student layers/attention map"
    )
    selection_indices: Optional[List[int]] = Field(
        default=None,
        description="If selected strategy `select`, provides mapping between student and teacher model layer/attention map",
        examples=[[1,2,0]] # mapping 0th student layer to 1st layer of teacher and so on
    )
    weights: Optional[List[List[int]]] = Field(
        default=None,
        description="If selected strategy `weighted`, provides the weights for each of the teacher layers to be used when computing student layer",
        examples=[[[100,10,40,50], [10,5,50,25]]] # 0th of student is computed with teacher layer weights 100 for 0th layer, 10 for 1st layer and so on
    )

    remove_unused_columns: Optional[bool] = Field(
        default=False,
    )

    model_config = {
        "extra": "forbid"
    }