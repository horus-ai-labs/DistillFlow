from dataclasses import dataclass, field
from typing import Literal, List, Optional

@dataclass
class FreezeArguments:
    r"""
    Arguments pertaining to the freeze (partial-parameter) training.
    """

    freeze_trainable_layers: int = field(
        default=2,
        metadata={
            "help": (
                "The number of trainable layers for freeze (partial-parameter) fine-tuning. "
                "Positive numbers mean the last n layers are set as trainable, "
                "negative numbers mean the first n layers are set as trainable."
            )
        },
    )
    freeze_trainable_modules: str = field(
        default="all",
        metadata={
            "help": (
                "Name(s) of trainable modules for freeze (partial-parameter) fine-tuning. "
                "Use commas to separate multiple modules. "
                "Use `all` to specify all the available modules."
            )
        },
    )
    freeze_extra_modules: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Name(s) of modules apart from hidden layers to be set as trainable "
                "for freeze (partial-parameter) fine-tuning. "
                "Use commas to separate multiple modules."
            )
        },
    )


@dataclass
class LoraArguments:
    r"""
    Arguments pertaining to the LoRA training.
    """

    additional_target: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Name(s) of modules apart from LoRA layers to be set as trainable "
                "and saved in the final checkpoint. "
                "Use commas to separate multiple modules."
            )
        },
    )
    lora_alpha: Optional[int] = field(
        default=None,
        metadata={"help": "The scale factor for LoRA fine-tuning (default: lora_rank * 2)."},
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout rate for the LoRA fine-tuning."},
    )
    lora_rank: int = field(
        default=8,
        metadata={"help": "The intrinsic dimension for LoRA fine-tuning."},
    )
    lora_target: str = field(
        default="all",
        metadata={
            "help": (
                "Name(s) of target modules to apply LoRA. "
                "Use commas to separate multiple modules. "
                "Use `all` to specify all the linear modules."
            )
        },
    )
    loraplus_lr_ratio: Optional[float] = field(
        default=None,
        metadata={"help": "LoRA plus learning rate ratio (lr_B / lr_A)."},
    )
    loraplus_lr_embedding: float = field(
        default=1e-6,
        metadata={"help": "LoRA plus learning rate for lora embedding layers."},
    )
    use_rslora: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the rank stabilization scaling factor for LoRA layer."},
    )
    use_dora: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the weight-decomposed lora method (DoRA)."},
    )
    pissa_init: bool = field(
        default=False,
        metadata={"help": "Whether or not to initialize a PiSSA adapter."},
    )
    pissa_iter: int = field(
        default=16,
        metadata={"help": "The number of iteration steps performed by FSVD in PiSSA. Use -1 to disable it."},
    )
    pissa_convert: bool = field(
        default=False,
        metadata={"help": "Whether or not to convert the PiSSA adapter to a normal LoRA adapter."},
    )
    create_new_adapter: bool = field(
        default=False,
        metadata={"help": "Whether or not to create a new adapter with randomly initialized weight."},
    )

@dataclass
class FinetuningArguments(FreezeArguments, LoraArguments):
    r"""
    Arguments pertaining to which techniques we are going to fine-tuning with.
    """

    pure_bf16: bool = field(
        default=False,
        metadata={"help": "Whether or not to train model in purely bf16 precision (without AMP)."},
    )
    stage: Literal["pt", "sft", "rm", "ppo", "dpo", "kto"] = field(
        default="sft",
        metadata={"help": "Which stage will be performed in training."},
    )
    finetuning_type: Literal["lora", "freeze", "full"] = field(
        default="lora",
        metadata={"help": "Which fine-tuning method to use."},
    )
    use_llama_pro: bool = field(
        default=False,
        metadata={"help": "Whether or not to make only the parameters in the expanded blocks trainable."},
    )
    use_adam_mini: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the Adam-mini optimizer."},
    )
    freeze_vision_tower: bool = field(
        default=True,
        metadata={"help": "Whether ot not to freeze vision tower in MLLM training."},
    )
    train_mm_proj_only: bool = field(
        default=False,
        metadata={"help": "Whether or not to train the multimodal projector for MLLM only."},
    )
    compute_accuracy: bool = field(
        default=False,
        metadata={"help": "Whether or not to compute the token-level accuracy at evaluation."},
    )
    plot_loss: bool = field(
        default=False,
        metadata={"help": "Whether or not to save the training loss curves."},
    )

    def __post_init__(self):
        def split_arg(arg):
            if isinstance(arg, str):
                return [item.strip() for item in arg.split(",")]
            return arg

        # self.freeze_trainable_modules: List[str] = split_arg(self.freeze_trainable_modules)
        # self.freeze_extra_modules: Optional[List[str]] = split_arg(self.freeze_extra_modules)
        # self.lora_alpha: int = self.lora_alpha or self.lora_rank * 2
        # self.lora_target: List[str] = split_arg(self.lora_target)
        # self.additional_target: Optional[List[str]] = split_arg(self.additional_target)
        # self.galore_target: List[str] = split_arg(self.galore_target)
        self.freeze_vision_tower = self.freeze_vision_tower or self.train_mm_proj_only
        # self.use_ref_model = self.stage == "dpo" and self.pref_loss not in ["orpo", "simpo"]

        assert self.finetuning_type in ["lora", "freeze", "full"], "Invalid fine-tuning method."
        # assert self.ref_model_quantization_bit in [None, 8, 4], "We only accept 4-bit or 8-bit quantization."
        # assert self.reward_model_quantization_bit in [None, 8, 4], "We only accept 4-bit or 8-bit quantization."

        # if self.stage == "ppo" and self.reward_model is None:
        #     raise ValueError("`reward_model` is necessary for PPO training.")

        # if self.stage == "ppo" and self.reward_model_type == "lora" and self.finetuning_type != "lora":
        #     raise ValueError("`reward_model_type` cannot be lora for Freeze/Full PPO training.")

        # if self.stage == "dpo" and self.pref_loss != "sigmoid" and self.dpo_label_smoothing > 1e-6:
        #     raise ValueError("`dpo_label_smoothing` is only valid for sigmoid loss function.")

        if self.use_llama_pro and self.finetuning_type == "full":
            raise ValueError("`use_llama_pro` is only valid for Freeze or LoRA training.")


        if self.train_mm_proj_only and self.finetuning_type != "full":
            raise ValueError("`train_mm_proj_only` is only valid for full training.")
