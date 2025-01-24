from pydantic import BaseModel, Field, model_validator

from typing import Optional, Literal, List

from .template.args import TemplateArgs

class DatasetArgs(BaseModel):
    path: str = Field(
        description="The path to huggingface dataset"
    )
    template: TemplateArgs = None
    num_samples: Optional[int] = Field (
        default=None,
        description="Number of samples to pick from the given dataset",
        examples=[1000, 100_000]
    )
    load_from_cache_file: bool = Field(
        default=True,
        description="Should load the dataset from cache if available, if disabled, the dataset will be re-synced from HF"
    )
    split: str = Field(
        default="train",
        description="Split to be used for this dataset (defaults to `train`)",
        examples=["train", "test"]
    )

    model_config = {
        "extra": "forbid"
    }

class DataArgs(BaseModel):
    r"""
    Arguments pertaining to what data we are going to input our model for training and evaluation.
    """
    seed: Optional[int] = Field(
        default=0,
        description="Seed to use when shuffling the dataset."
    )
    train_datasets: List[DatasetArgs] = Field(
        description="The dataset(s) to use for training. Provide as a list of DatasetArgs."
    )
    eval_datasets: Optional[List[DatasetArgs]] = Field(
        default=None,
        description="The names of dataset(s) to use for evaluation. Provide as a list of DatasetArgs."
    )
    cache_dir: Optional[str] = Field(
        default=None,
        description="Where to store the pre-trained datasets downloaded."
    )
    hf_hub_token: Optional[str] = Field(
        default=None,
        description="Auth token to log in with Hugging Face Hub."
    )
    text_field: Optional[str] = Field(
        default=None,
        description="Name of the field key to convert the dataset."
    )
    streaming: bool = Field(
        default=False,
        description="Enable dataset streaming."
    )
    buffer_size: Optional[int] = Field(
        default=16384,
        description="Size of the buffer to randomly sample examples from in dataset streaming."
    )
    mix_strategy: Optional[Literal["concat", "interleave_under", "interleave_over"]] = Field(
        default="concat",
        description="Strategy to use in dataset mixing (concat/interleave) (undersampling/oversampling)."
    )
    interleave_probs: Optional[str] = Field(
        default=None,
        description="Probabilities to sample data from datasets. Use commas to separate multiple datasets."
    )
    max_samples: Optional[int] = Field(
        default=None,
        description="For debugging purposes, truncate the number of examples for each dataset."
    )
    test_size: float = Field(
        default=0.0,
        description="Size of the development set, should be an integer or a float in range `[0,1)`.",
        examples=[1000, 100_000, 0.2, 0.5]
    )

    @model_validator(mode='after')
    def validate_args(self) -> 'DataArgs':
        if self.eval_datasets is not None and self.test_size > 0:
            raise ValueError("Cannot specify `val_size` if `eval_dataset` is not None.")

        if self.streaming and 1e-6 < self.test_size < 1:
            raise ValueError("Streaming mode should have an integer test size.")

        if self.streaming and self.max_samples is not None:
            raise ValueError("`max_samples` is incompatible with `streaming`.")

        return self

    model_config = {
        "extra": "forbid"
    }
