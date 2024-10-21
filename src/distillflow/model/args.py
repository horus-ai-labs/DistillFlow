from dataclasses import field, dataclass
from typing import Optional, Literal, Union, Dict, Any

import torch

@dataclass
class QuantizationArguments:
    r"""
    Arguments pertaining to the quantization method.
    """

    quantization_method: Literal["bitsandbytes", "hqq", "eetq"] = field(
        default="bitsandbytes",
        metadata={"help": "Quantization method to use for on-the-fly quantization."},
    )
    quantization_bit: Optional[int] = field(
        default=None,
        metadata={"help": "The number of bits to quantize the model using on-the-fly quantization."},
    )
    quantization_type: Literal["fp4", "nf4"] = field(
        default="nf4",
        metadata={"help": "Quantization data type to use in bitsandbytes int4 training."},
    )
    double_quantization: bool = field(
        default=True,
        metadata={"help": "Whether or not to use double quantization in bitsandbytes int4 training."},
    )
    quantization_device_map: Optional[Literal["auto"]] = field(
        default=None,
        metadata={"help": "Device map used to infer the 4-bit quantized model, needs bitsandbytes>=0.43.0."},
    )

@dataclass
class ModelArguments(QuantizationArguments):
    r"""
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune or infer.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the model weight or identifier from huggingface.co/models or modelscope.cn/models."
        },
    )
    adapter_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to the adapter weight or identifier from huggingface.co/models. "
                "Use commas to separate multiple adapters."
            )
        },
    )
    adapter_folder: Optional[str] = field(
        default=None,
        metadata={"help": "The folder containing the adapter weights to load."},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pre-trained models downloaded from huggingface.co or modelscope.cn."},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether or not to use one of the fast tokenizer (backed by the tokenizers library)."},
    )
    resize_vocab: bool = field(
        default=False,
        metadata={"help": "Whether or not to resize the tokenizer vocab and the embedding layers."},
    )
    split_special_tokens: bool = field(
        default=False,
        metadata={"help": "Whether or not the special tokens should be split during the tokenization process."},
    )
    new_special_tokens: Optional[str] = field(
        default=None,
        metadata={"help": "Special tokens to be added into the tokenizer. Use commas to separate multiple tokens."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    # low_cpu_mem_usage: bool = field(
    #     default=True,
    #     metadata={"help": "Whether or not to use memory-efficient model loading."},
    # )
    # rope_scaling: Optional[Literal["linear", "dynamic"]] = field(
    #     default=None,
    #     metadata={"help": "Which scaling strategy should be adopted for the RoPE embeddings."},
    # )
    # flash_attn: Literal["auto", "disabled", "sdpa", "fa2"] = field(
    #     default="auto",
    #     metadata={"help": "Enable FlashAttention for faster training and inference."},
    # )
    # shift_attn: bool = field(
    #     default=False,
    #     metadata={"help": "Enable shift short attention (S^2-Attn) proposed by LongLoRA."},
    # )
    # mixture_of_depths: Optional[Literal["convert", "load"]] = field(
    #     default=None,
    #     metadata={"help": "Convert the model to mixture-of-depths (MoD) or load the MoD model."},
    # )
    use_unsloth: bool = field(
        default=False,
        metadata={"help": "Whether or not to use unsloth's optimization for the LoRA training."},
    )
    # use_unsloth_gc: bool = field(
    #     default=False,
    #     metadata={"help": "Whether or not to use unsloth's gradient checkpointing."},
    # )
    # enable_liger_kernel: bool = field(
    #     default=False,
    #     metadata={"help": "Whether or not to enable liger kernel for faster training."},
    # )
    # moe_aux_loss_coef: Optional[float] = field(
    #     default=None,
    #     metadata={"help": "Coefficient of the auxiliary router loss in mixture-of-experts model."},
    # )
    # disable_gradient_checkpointing: bool = field(
    #     default=False,
    #     metadata={"help": "Whether or not to disable gradient checkpointing."},
    # )
    # upcast_layernorm: bool = field(
    #     default=False,
    #     metadata={"help": "Whether or not to upcast the layernorm weights in fp32."},
    # )
    # upcast_lmhead_output: bool = field(
    #     default=False,
    #     metadata={"help": "Whether or not to upcast the output of lm_head in fp32."},
    # )
    # train_from_scratch: bool = field(
    #     default=False,
    #     metadata={"help": "Whether or not to randomly initialize the model weights."},
    # )
    # infer_backend: Literal["huggingface", "vllm"] = field(
    #     default="huggingface",
    #     metadata={"help": "Backend engine used at inference."},
    # )
    offload_folder: str = field(
        default="offload",
        metadata={"help": "Path to offload model weights."},
    )
    # use_cache: bool = field(
    #     default=True,
    #     metadata={"help": "Whether or not to use KV cache in generation."},
    # )
    infer_dtype: Literal["auto", "float16", "bfloat16", "float32"] = field(
        default="auto",
        metadata={"help": "Data type for model weights and activations at inference."},
    )
    hf_hub_token: Optional[str] = field(
        default=None,
        metadata={"help": "Auth token to log in with Hugging Face Hub."},
    )
    # ms_hub_token: Optional[str] = field(
    #     default=None,
    #     metadata={"help": "Auth token to log in with ModelScope Hub."},
    # )
    # om_hub_token: Optional[str] = field(
    #     default=None,
    #     metadata={"help": "Auth token to log in with Modelers Hub."},
    # )
    print_param_status: bool = field(
        default=False,
        metadata={"help": "For debugging purposes, print the status of the parameters in the model."},
    )
    compute_dtype: Optional[torch.dtype] = field(
        default=None,
        init=False,
        metadata={"help": "Torch data type for computing model outputs, derived from `fp/bf16`. Do not specify it."},
    )
    device_map: Optional[Union[str, Dict[str, Any]]] = field(
        default=None,
        init=False,
        metadata={"help": "Device map for model placement, derived from training stage. Do not specify it."},
    )
    model_max_length: Optional[int] = field(
        default=None,
        init=False,
        metadata={"help": "The maximum input length for model, derived from `cutoff_len`. Do not specify it."},
    )
    block_diag_attn: bool = field(
        default=False,
        init=False,
        metadata={"help": "Whether use block diag attention or not, derived from `neat_packing`. Do not specify it."},
    )