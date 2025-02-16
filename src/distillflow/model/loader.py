import math
import os
from contextlib import nullcontext
from types import MethodType
from typing import Dict, Any

from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import PreTrainedTokenizer, AutoConfig, PreTrainedModel, \
    PretrainedConfig, AutoModelForCausalLM, is_torch_npu_available
from transformers.integrations import is_deepspeed_zero3_enabled, is_deepspeed_available
from transformers.modeling_utils import is_fsdp_enabled
from transformers.utils import is_torch_sdpa_available, is_flash_attn_2_available
from transformers.utils.versions import require_version

from .adapter import init_adapter
from .checkpoint import prepare_model_for_training
from .liger_kernel import apply_liger_kernel
from .quantization import _configure_quantization, QuantizationMethod
from .tokenizer import load_tokenizer
from .unsloth import load_unsloth_pretrained_model
from ..common.logger import get_logger
from .args import ModelArgs
import torch

from ..common import count_parameters, infer_optim_dtype, get_current_device

logger = get_logger(__name__)

def get_init_kwargs(model_args: ModelArgs) -> Dict[str, Any]:
    r"""
    Gets arguments to load config/tokenizer/model.

    Note: including inplace operation of model_args.
    """
    return {
        "trust_remote_code": True,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token
    }

def _register_autoclass(config: PretrainedConfig, model: "PreTrainedModel", tokenizer: "PreTrainedTokenizer"):
    if "AutoConfig" in getattr(config, "auto_map", {}):
        config.__class__.register_for_auto_class()
    if "AutoModelForCausalLM" in getattr(config, "auto_map", {}):
        model.__class__.register_for_auto_class()
    if "AutoTokenizer" in tokenizer.init_kwargs.get("auto_map", {}):
        tokenizer.__class__.register_for_auto_class()

def _configure_attn_implementation(
        config: PretrainedConfig, model_args: ModelArgs, is_trainable: bool
) -> None:
    if getattr(config, "model_type", None) == "gemma2" and is_trainable:
        if model_args.flash_attn == "auto" or model_args.flash_attn == "fa2":
            if is_flash_attn_2_available():
                require_version("flash_attn>=2.6.3", "To fix: pip install flash_attn>=2.6.3")
                if model_args.flash_attn != "fa2":
                    logger.warning("Gemma-2 should use flash attention 2, change `flash_attn` to fa2.")
                    model_args.flash_attn = "fa2"
            else:
                logger.warning("FlashAttention-2 is not installed, use eager attention.")
                model_args.flash_attn = "disabled"
        elif model_args.flash_attn == "sdpa":
            logger.warning("Gemma-2 should use soft-capping attention, while the SDPA attention does not support it.")

    if model_args.flash_attn == "auto":
        return

    elif model_args.flash_attn == "disabled":
        requested_attn_implementation = "eager"

    elif model_args.flash_attn == "sdpa":
        if not is_torch_sdpa_available():
            logger.warning("torch>=2.1.1 is required for SDPA attention.")
            return

        requested_attn_implementation = "sdpa"
    elif model_args.flash_attn == "fa2":
        if not is_flash_attn_2_available():
            logger.warning("FlashAttention-2 is not installed.")
            return

        requested_attn_implementation = "flash_attention_2"
    else:
        raise NotImplementedError("Unknown attention type: {}".format(model_args.flash_attn))

    if getattr(config, "model_type", None) == "internlm2":  # special case for custom models
        setattr(config, "attn_implementation", requested_attn_implementation)
    else:
        setattr(config, "_attn_implementation", requested_attn_implementation)

def _patch_config(
        config: PretrainedConfig,
        tokenizer: PreTrainedTokenizer,
        model_args: ModelArgs,
        init_kwargs: Dict[str, Any],
        is_trainable: bool,
        torch_dtype: torch.dtype
) -> None:
    if is_torch_npu_available():
        use_jit_compile = os.environ.get("JIT_COMPILE", "0").lower() in ["true", "1"]
        torch.npu.set_compile_mode(jit_compile=use_jit_compile)

    _configure_attn_implementation(config, model_args, is_trainable)
    _configure_quantization(config, tokenizer, model_args.quantization_args, init_kwargs, torch_dtype)

    if model_args.use_cache and not is_trainable:
        setattr(config, "use_cache", True)
        logger.info("Using KV cache for faster generation.")

    if getattr(config, "model_type", None) == "qwen":
        setattr(config, "use_flash_attn", model_args.flash_attn == "fa2")
        for dtype_name, dtype in [("fp16", torch.float16), ("bf16", torch.bfloat16), ("fp32", torch.float32)]:
            setattr(config, dtype_name, torch_dtype == dtype)

    if getattr(config, "model_type", None) == "qwen2" and is_trainable and model_args.flash_attn == "fa2":
        setattr(config, "use_cache", False)  # qwen2 does not support use_cache when using flash attn

    # if "LlavaLlamaForCausalLM" in getattr(config, "architectures", []):
    #     raise ValueError("Please download llava models with hf-compatible format: https://huggingface.co/llava-hf")

    # deepspeed zero3 is not compatible with low_cpu_mem_usage

    init_kwargs["low_cpu_mem_usage"] = model_args.low_cpu_mem_usage and (not is_deepspeed_available())

    # cast data type of the model if:
    # 1. not deepspeed zero3 and not fsdp (keep zero3 or fsdp in float32)
    # 2. quantization_bit is not None (qlora)

    quantization_args = model_args.quantization_args
    if (not is_deepspeed_zero3_enabled() and not is_fsdp_enabled()) or quantization_args.quantization_bit is not None:
        init_kwargs["torch_dtype"] = torch_dtype

        if init_kwargs["low_cpu_mem_usage"]:  # device map requires low_cpu_mem_usage=True
            if "device_map" not in init_kwargs:
                init_kwargs["device_map"] = {"": get_current_device()}

            if init_kwargs.get("device_map", None) == "auto":
                init_kwargs["offload_folder"] = model_args.offload_folder


def _noisy_mean_initialization(embed_weight: "torch.Tensor", num_new_tokens: int) -> None:
    embedding_dim = embed_weight.size(1)
    avg_weight = embed_weight[:-num_new_tokens].mean(dim=0, keepdim=True)
    noise_weight = torch.empty_like(embed_weight[-num_new_tokens:])
    noise_weight.normal_(mean=0, std=(1.0 / math.sqrt(embedding_dim)))
    embed_weight[-num_new_tokens:] = avg_weight + noise_weight

def _resize_embedding_layer(model: "PreTrainedModel", tokenizer: "PreTrainedTokenizer") -> None:
    r"""
    Resize token embeddings.
    """
    # if is_deepspeed_zero3_enabled():
    #     import deepspeed  # type: ignore
    #
    #     params = [model.get_input_embeddings().weight]
    #     if model.get_output_embeddings() is not None and not model.config.tie_word_embeddings:
    #         params.append(model.get_output_embeddings().weight)
    #
    #     context_maybe_zero3 = deepspeed.zero.GatheredParameters(params, modifier_rank=0)
    # else:
    context_maybe_zero3 = nullcontext()

    with context_maybe_zero3:
        current_embedding_size = model.get_input_embeddings().weight.size(0)

    if len(tokenizer) > current_embedding_size:
        if getattr(model, "quantization_method", None):
            raise ValueError("Cannot resize embedding layers of a quantized model.")

        if not isinstance(model.get_output_embeddings(), torch.nn.Linear):
            raise ValueError("Current model does not support resizing embedding layers.")

        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
        with context_maybe_zero3:
            new_embedding_size = model.get_input_embeddings().weight.size(0)
            num_new_tokens = new_embedding_size - current_embedding_size
            _noisy_mean_initialization(model.get_input_embeddings().weight.data, num_new_tokens)
            _noisy_mean_initialization(model.get_output_embeddings().weight.data, num_new_tokens)

        logger.info("Resized token embeddings from {} to {}.".format(current_embedding_size, new_embedding_size))

def _patch_model(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        model_args: ModelArgs,
        is_trainable: bool,
) -> None:
    gen_config = model.generation_config  # check and fix generation config
    if not gen_config.do_sample and (
            (gen_config.temperature is not None and gen_config.temperature != 1.0)
            or (gen_config.top_p is not None and gen_config.top_p != 1.0)
            or (gen_config.typical_p is not None and gen_config.typical_p != 1.0)
    ):
        gen_config.do_sample = True

    if "GenerationMixin" not in str(model.generate.__func__):
        model.generate = MethodType(PreTrainedModel.generate, model)

    if model_args.resize_vocab:
        _resize_embedding_layer(model, tokenizer)

    if is_trainable:
        prepare_model_for_training(model, model_args)
        # autocast_projector_dtype(model, model_args)
        # add_z3_leaf_module(model)

    attn = getattr(model.config, "_attn_implementation", None)
    if attn == "flash_attention_2":
        logger.info("Using FlashAttention-2 for faster training and inference.")
    elif attn == "sdpa":
        logger.info("Using torch SDPA for faster training and inference.")

def load_model(
        model_args: ModelArgs,
        is_trainable: bool = False,
) -> (PreTrainedModel, PreTrainedTokenizer):
    tokenizer = load_tokenizer(model_args)
    init_kwargs = get_init_kwargs(model_args)
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)
    if model_args.infer_dtype != "auto" and not is_trainable:
        torch_dtype = getattr(torch, model_args.infer_dtype)
    else:
        torch_dtype = infer_optim_dtype(model_dtype=getattr(config, "torch_dtype", None))

    _patch_config(config, tokenizer, model_args, init_kwargs, is_trainable, torch_dtype)

    # More details here: https://github.com/linkedin/Liger-Kernel
    apply_liger_kernel(config, model_args, is_trainable, require_logits=True)

    model = None
    lazy_load = False
    if model_args.use_unsloth:
        if model_args.adapter_name_or_path is not None:
            lazy_load = True
        elif is_trainable:
            model = load_unsloth_pretrained_model(config, model_args)

    quantization_args = model_args.quantization_args
    if quantization_args.quantization_method == QuantizationMethod.GPTQ.value:
        quantize_config = BaseQuantizeConfig(
            bits=quantization_args.quantization_bit,
            group_size=128,       # Group size (optional, can be None)
            desc_act=False        # Disable activation descriptor (optional)
        )
        model = AutoGPTQForCausalLM.from_pretrained(model_args.model_name_or_path, bits=quantization_args.quantization_bit, group_size=128, quantize_config=quantize_config)

    if model is None and not lazy_load:
        init_kwargs["config"] = config
        init_kwargs["pretrained_model_name_or_path"] = model_args.model_name_or_path
        model = AutoModelForCausalLM.from_pretrained(**init_kwargs)

    if not lazy_load:
        _patch_model(model, tokenizer, model_args, is_trainable)
        _register_autoclass(config, model, tokenizer)

    model = init_adapter(config, model, model_args, is_trainable)
    if not is_trainable:
        model.requires_grad_(False)
        for param in model.parameters():
            if param.data.dtype == torch.float32 and torch_dtype != torch.float32:
                param.data = param.data.to(torch_dtype)

        model.eval()
    else:
        model.train()

    trainable_params, all_param = count_parameters(model)
    if is_trainable:
        param_stats = "trainable params: {:,} || all params: {:,} || trainable%: {:.4f}".format(
            trainable_params, all_param, 100 * trainable_params / all_param
        )
    else:
        param_stats = "all params: {:,}".format(all_param)

    logger.info(param_stats)

    if model_args.print_param_status:
        for name, param in model.named_parameters():
            print(
                "name: {}, dtype: {}, device: {}, trainable: {}".format(
                    name, param.dtype, param.device, param.requires_grad
                )
            )

    return model, tokenizer