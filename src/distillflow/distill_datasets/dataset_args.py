from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence


@dataclass
class DatasetAttr:
    dataset_name: str
    path: str = ""
    formatting: Literal["alpaca", "sharegpt"] = "alpaca"
    # ranking: bool = False
    split: str = "train"
