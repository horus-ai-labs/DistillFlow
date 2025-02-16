from typing import Literal, List, Optional

from pydantic import Field, BaseModel


class FreezeArgs(BaseModel):
    r"""
    Arguments pertaining to the freeze (partial-parameter) training.
    """

    freeze_trainable_layers: int = Field(
        default=2,
        description=(
                "The number of trainable layers for freeze (partial-parameter) fine-tuning. "
                "Positive numbers mean the last n layers are set as trainable, "
                "negative numbers mean the first n layers are set as trainable."
        )
    )
    freeze_trainable_modules: str = Field(
        default="all",
        description=(
                "Name(s) of trainable modules for freeze (partial-parameter) fine-tuning. "
                "Use commas to separate multiple modules. "
                "Use `all` to specify all the available modules."
        )
    )
    freeze_extra_modules: Optional[str] = Field(
        default=None,
        description=(
                "Name(s) of modules apart from hidden layers to be set as trainable "
                "for freeze (partial-parameter) fine-tuning. "
                "Use commas to separate multiple modules."
        )
    )


class LoraArgs(BaseModel):
    r"""
    Arguments pertaining to the LoRA training.
    """

    additional_target: Optional[str] = Field(
        default=None,
        description=(
                "Name(s) of modules apart from LoRA layers to be set as trainable "
                "and saved in the final checkpoint. "
                "Use commas to separate multiple modules."
        )
    )
    lora_alpha: Optional[int] = Field(
        default=None,
        description="The scale factor for LoRA fine-tuning (default: lora_rank * 2)."
    )
    lora_dropout: float = Field(
        default=0.0,
        description="Dropout rate for the LoRA fine-tuning."
    )
    lora_rank: int = Field(
        default=8,
        description="The intrinsic dimension for LoRA fine-tuning."
    )
    lora_target: str = Field(
        default="all",
        description=(
                "Name(s) of target modules to apply LoRA. "
                "Use commas to separate multiple modules. "
                "Use `all` to specify all the linear modules."
        )
    )
    loraplus_lr_ratio: Optional[float] = Field(
        default=None,
        description="LoRA plus learning rate ratio (lr_B / lr_A)."
    )
    loraplus_lr_embedding: float = Field(
        default=1e-6,
        description="LoRA plus learning rate for lora embedding layers."
    )
    use_rslora: bool = Field(
        default=False,
        description="Whether or not to use the rank stabilization scaling factor for LoRA layer."
    )
    use_dora: bool = Field(
        default=False,
        description="Whether or not to use the weight-decomposed lora method (DoRA)."
    )
    pissa_init: bool = Field(
        default=False,
        description="Whether or not to initialize a PiSSA adapter."
    )
    pissa_iter: int = Field(
        default=16,
        description="The number of iteration steps performed by FSVD in PiSSA. Use -1 to disable it."
    )
    pissa_convert: bool = Field(
        default=False,
        description="Whether or not to convert the PiSSA adapter to a normal LoRA adapter."
    )
    create_new_adapter: bool = Field(
        default=False,
        description="Whether or not to create a new adapter with randomly initialized weight."
    )


class FinetuningArgs(FreezeArgs, LoraArgs, BaseModel):
    r"""
    Arguments pertaining to which techniques we are going to fine-tuning with.
    """

    finetuning_type: Literal["lora", "freeze", "full"] = Field(
        default="full",
        description="Which fine-tuning method to use."
    )
    use_llama_pro: bool = Field(
        default=False,
        description="Whether or not to make only the parameters in the expanded blocks trainable."
    )

    def __post_init__(self):
        def split_arg(arg):
            if isinstance(arg, str):
                return [item.strip() for item in arg.split(",")]
            return arg

        self.freeze_trainable_modules: List[str] = split_arg(self.freeze_trainable_modules)
        self.freeze_extra_modules: Optional[List[str]] = split_arg(self.freeze_extra_modules)
        self.lora_alpha: int = self.lora_alpha or self.lora_rank * 2
        self.lora_target: List[str] = split_arg(self.lora_target)
        self.additional_target: Optional[List[str]] = split_arg(self.additional_target)

        assert self.finetuning_type in ["lora", "freeze", "full"], "Invalid fine-tuning method."
        if self.use_llama_pro and self.finetuning_type == "full":
            raise ValueError("`use_llama_pro` is only valid for Freeze or LoRA training.")
