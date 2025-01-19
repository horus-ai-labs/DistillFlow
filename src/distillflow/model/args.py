from typing import Optional, Literal

from pydantic import Field, BaseModel

class TokenizerArgs(BaseModel):
    template: Optional[str] = Field(
        default=None,
        description="Template to use "
    )

class QuantizationArgs(BaseModel):
    r"""
    Arguments pertaining to the quantization method.
    """

    quantization_method: Literal["bitsandbytes", "hqq", "eetq", "gptq"] = Field(
        default="bitsandbytes",
        description="Quantization method to use for on-the-fly quantization."
    )
    quantization_bit: Optional[int] = Field(
        default=None,
        description="The number of bits to quantize the model using on-the-fly quantization."
    )
    quantization_type: Literal["fp4", "nf4"] = Field(
        default="nf4",
        description="Quantization data type to use in bitsandbytes int4 training."
    )
    double_quantization: bool = Field(
        default=True,
        description="Whether or not to use double quantization in bitsandbytes int4 training."
    )
    quantization_device_map: Optional[Literal["auto"]] = Field(
        default=None,
        description="Device map used to infer the 4-bit quantized model, needs bitsandbytes>=0.43.0."
    )

    model_config = {
        "extra": "forbid"
    }

class ModelArgs(BaseModel):
    r"""
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune or infer.
    """

    model_name_or_path: str = Field(
        description="Path to the model weight or identifier from huggingface.co/models or modelscope.cn/models.",
        examples=["Qwen/Qwen2-0.5B"]
    )
    adapter_name_or_path: Optional[str] = Field(
        default=None,
        description="Path to the adapter weight or identifier from huggingface.co/models. "
                "Use commas to separate multiple adapters."
    )
    adapter_folder: Optional[str] = Field(
        default=None,
        description="The folder containing the adapter weights to load."
    )
    cache_dir: Optional[str] = Field(
        default=None,
        description="Where to store the pre-trained models downloaded from huggingface.co or modelscope.cn."
    )
    resize_vocab: bool = Field(
        default=False,
        description="Whether or not to resize the tokenizer vocab and the embedding layers."
    )
    split_special_tokens: bool = Field(
        default=False,
        description="Whether or not the special tokens should be split during the tokenization process."
    )
    new_special_tokens: Optional[str] = Field(
        default=None,
        description="Special tokens to be added into the tokenizer. Use commas to separate multiple tokens."
    )
    model_revision: str = Field(
        default="main",
        description="The specific model version to use (can be a branch name, tag name or commit id)."
    )
    low_cpu_mem_usage: bool = Field(
        default=True,
        description="Whether or not to use memory-efficient model loading."
    )
    flash_attn: Literal["auto", "disabled", "sdpa", "fa2"] = Field(
        default="auto",
        description="Enable FlashAttention for faster training and inference."
    )
    use_unsloth: bool = Field(
        default=False,
        description="Whether or not to use unsloth's optimization for the LoRA training."
    )
    use_unsloth_gc: bool = Field(
        default=False,
        description="Whether or not to use unsloth's gradient checkpointing."
    )
    enable_liger_kernel: bool = Field(
        default=False,
        description="Whether or not to enable liger kernel for faster training."
    )
    disable_gradient_checkpointing: bool = Field(
        default=False,
        description="Whether or not to disable gradient checkpointing."
    )
    upcast_layernorm: bool = Field(
        default=False,
        description="Whether or not to upcast the layernorm weights in fp32."
    )
    upcast_lmhead_output: bool = Field(
        default=False,
        description="Whether or not to upcast the output of lm_head in fp32."
    )
    offload_folder: str = Field(
        default="offload",
        description="Path to offload model weights."
    )
    use_cache: bool = Field(
        default=True,
        description="Whether or not to use KV cache in generation."
    )
    infer_dtype: Literal["auto", "float16", "bfloat16", "float32"] = Field(
        default="auto",
        description="Data type for model weights and activations at inference."
    )
    hf_hub_token: Optional[str] = Field(
        default=None,
        description="Auth token to log in with Hugging Face Hub."
    )
    print_param_status: bool = Field(
        default=False,
        description="For debugging purposes, print the status of the parameters in the model."
    )
    output_attentions: bool = Field(
        default=False,
        description="Whether to output the attention mask for the model during forward pass"
    )
    quantization_args: Optional[QuantizationArgs] = Field(
        default=QuantizationArgs(),
        description="Arguments related to quantization"
    )

    model_config = {
        "extra": "forbid"
    }