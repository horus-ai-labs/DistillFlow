from dataclasses import dataclass
from enum import Enum, unique
from typing import Literal, Optional, Sequence

@unique
class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    OBSERVATION = "observation"

@dataclass
class DatasetArgs:
    # dataset_name: str
    # load_from: Literal["hf_hub", "ms_hub", "om_hub", "script", "file"]

    path: str = ""
    num_samples: Optional[int] = None
    to_text: bool = False

    # formatting: Literal["alpaca", "sharegpt"] = "alpaca"
    # ranking: bool = False
    split: str = "train"
    # system: Optional[str] = None
    # tools: Optional[str] = None

@dataclass
class DataArgs:
    train_dataset: DatasetArgs
    eval_dataset: Optional[DatasetArgs] = None

    val_size: float = 0.0 # Size of the development set, should be an integer or a float in range `[0,1)`