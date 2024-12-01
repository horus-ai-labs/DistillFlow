import math
import os
from contextlib import nullcontext
from types import MethodType
from typing import TypedDict, Optional, Dict, Any

from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
# from auto_gptq import AutoGPTQForCausalLM
from transformers import PreTrainedTokenizer, ProcessorMixin, AutoConfig, AutoTokenizer, AutoProcessor, PreTrainedModel, \
    PretrainedConfig, AutoModelForCausalLM, GPTQConfig, is_torch_npu_available
from transformers.utils import is_torch_sdpa_available, is_flash_attn_2_available
from transformers.utils.versions import require_version

from .adapter import init_adapter
from .checkpoint import prepare_model_for_training
from .finetuning_args import FinetuningArguments
from .quantization import configure_quantization, QuantizationMethod
from .unsloth import load_unsloth_pretrained_model
from ..misc.logger import get_logger
from .args import ModelArguments
import torch

from ..misc import count_parameters, infer_optim_dtype
from transformers.models.llama.modeling_llama import (
    Cache,
    LlamaAttention,
    LlamaFlashAttention2,
    LlamaSdpaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
)

logger = get_logger(__name__)

def _get_init_kwargs(model_args: ModelArguments) -> Dict[str, Any]:
    r"""
    Gets arguments to load config/tokenizer/model.

    Note: including inplace operation of model_args.
    """
    # skip_check_imports()
    # model_args.model_name_or_path = try_download_model_from_other_hub(model_args)
    return {
        "trust_remote_code": True,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token,
    }

class TokenizerModule(TypedDict):
    tokenizer: PreTrainedTokenizer
    processor: Optional[ProcessorMixin]

def load_tokenizer(model_args: ModelArguments) -> TokenizerModule:
    r"""
    Loads pretrained tokenizer and optionally loads processor.

    Note: including inplace operation of model_args.
    """
    init_kwargs = _get_init_kwargs(model_args)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            split_special_tokens=model_args.split_special_tokens,
            padding_side="right",
            **init_kwargs,
        )
    except ValueError:  # try the fast one
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=True,
            padding_side="right",
            **init_kwargs,
        )
    except Exception as e:
        raise OSError("Failed to load tokenizer.") from e

    if model_args.new_special_tokens is not None:
        num_added_tokens = tokenizer.add_special_tokens(
            dict(additional_special_tokens=model_args.new_special_tokens),
            replace_additional_special_tokens=False,
        )
        logger.info("Add {} to special tokens.".format(",".join(model_args.new_special_tokens)))
        if num_added_tokens > 0 and not model_args.resize_vocab:
            model_args.resize_vocab = True
            logger.warning("New tokens have been added, changed `resize_vocab` to True.")
    else:
        tokenizer.pad_token = tokenizer.eos_token

    # patch_tokenizer(tokenizer)
    try:
        processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, **init_kwargs)
        # config = AutoConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)
        # patch_processor(processor, config, tokenizer, model_args)
    except Exception as e:
        logger.warning("Processor was not found: {}.".format(e))
        processor = None

    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/auto/processing_auto.py#L324
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None

    return {"tokenizer": tokenizer, "processor": processor}

def register_autoclass(config: PretrainedConfig, model: "PreTrainedModel", tokenizer: "PreTrainedTokenizer"):
    if "AutoConfig" in getattr(config, "auto_map", {}):
        config.__class__.register_for_auto_class()
    if "AutoModelForCausalLM" in getattr(config, "auto_map", {}):
        model.__class__.register_for_auto_class()
    if "AutoTokenizer" in tokenizer.init_kwargs.get("auto_map", {}):
        tokenizer.__class__.register_for_auto_class()

def configure_attn_implementation(
        config: PretrainedConfig, model_args: ModelArguments, is_trainable: bool
) -> None:
    if getattr(config, "model_type", None) == "gemma2" and is_trainable:
        if model_args.flash_attn == "auto" or model_args.flash_attn == "fa2":
            if is_flash_attn_2_available():
                require_version("transformers>=4.42.4", "To fix: pip install transformers>=4.42.4")
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


def patch_config(
        config: PretrainedConfig,
        tokenizer: PreTrainedTokenizer,
        model_args: ModelArguments,
        init_kwargs: Dict[str, Any],
        is_trainable: bool,
) -> None:
    if model_args.compute_dtype is None:  # priority: bf16 > fp16 > fp32
        if model_args.infer_dtype != "auto" and not is_trainable:
            model_args.compute_dtype = getattr(torch, model_args.infer_dtype)
        else:
            model_args.compute_dtype = infer_optim_dtype(model_dtype=getattr(config, "torch_dtype", None))

    if is_torch_npu_available():
        use_jit_compile = os.environ.get("JIT_COMPILE", "0").lower() in ["true", "1"]
        torch.npu.set_compile_mode(jit_compile=use_jit_compile)

    configure_attn_implementation(config, model_args, is_trainable)
    # configure_rope(config, model_args, is_trainable)
    # configure_longlora(config, model_args, is_trainable)
    configure_quantization(config, tokenizer, model_args, init_kwargs)
    # configure_moe(config, model_args, is_trainable)
    # configure_visual_model(config)
    # configure_packing(config, model_args, is_trainable)

    if model_args.use_cache and not is_trainable:
        setattr(config, "use_cache", True)
        logger.info("Using KV cache for faster generation.")

    if getattr(config, "model_type", None) == "qwen":
        setattr(config, "use_flash_attn", model_args.flash_attn == "fa2")
        for dtype_name, dtype in [("fp16", torch.float16), ("bf16", torch.bfloat16), ("fp32", torch.float32)]:
            setattr(config, dtype_name, model_args.compute_dtype == dtype)

    if getattr(config, "model_type", None) == "qwen2" and is_trainable and model_args.flash_attn == "fa2":
        setattr(config, "use_cache", False)  # qwen2 does not support use_cache when using flash attn

    # if "LlavaLlamaForCausalLM" in getattr(config, "architectures", []):
    #     raise ValueError("Please download llava models with hf-compatible format: https://huggingface.co/llava-hf")

    # deepspeed zero3 is not compatible with low_cpu_mem_usage
    init_kwargs["low_cpu_mem_usage"] = model_args.low_cpu_mem_usage # and (not is_deepspeed_zero3_enabled())

    # cast data type of the model if:
    # 1. not deepspeed zero3 and not fsdp (keep zero3 or fsdp in float32)
    # 2. quantization_bit is not None (qlora)
    if model_args.quantization_bit is not None:
        init_kwargs["torch_dtype"] = model_args.compute_dtype

        if init_kwargs["low_cpu_mem_usage"]:  # device map requires low_cpu_mem_usage=True
            if "device_map" not in init_kwargs and model_args.device_map:
                init_kwargs["device_map"] = model_args.device_map

            if init_kwargs.get("device_map", None) == "auto":
                init_kwargs["offload_folder"] = model_args.offload_folder


def _noisy_mean_initialization(embed_weight: "torch.Tensor", num_new_tokens: int) -> None:
    embedding_dim = embed_weight.size(1)
    avg_weight = embed_weight[:-num_new_tokens].mean(dim=0, keepdim=True)
    noise_weight = torch.empty_like(embed_weight[-num_new_tokens:])
    noise_weight.normal_(mean=0, std=(1.0 / math.sqrt(embedding_dim)))
    embed_weight[-num_new_tokens:] = avg_weight + noise_weight

def resize_embedding_layer(model: "PreTrainedModel", tokenizer: "PreTrainedTokenizer") -> None:
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

def patch_model(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        model_args: ModelArguments,
        is_trainable: bool,
        # add_valuehead: bool,
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

    # if add_valuehead:
    #     prepare_valuehead_model(model)

    if model_args.resize_vocab:
        resize_embedding_layer(model, tokenizer)

    if is_trainable:
        prepare_model_for_training(model, model_args)
        # autocast_projector_dtype(model, model_args)
        # add_z3_leaf_module(model)

    # if not model_args.use_unsloth:
    #     print_attn_implementation(model.config)

def load_model(
        tokenizer: PreTrainedTokenizer,
        model_args: "ModelArguments",
        finetuning_args: FinetuningArguments,
        is_trainable: bool = False,
        # add_valuehead: bool = False,
) -> PreTrainedModel:
    r"""
    Loads pretrained model.
    """
    init_kwargs = _get_init_kwargs(model_args)
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)
    configure_quantization(config, tokenizer, model_args, init_kwargs)
    # patch_config(config, tokenizer, model_args, init_kwargs, is_trainable)

    # TODO: Good optimization for huggingface models: https://github.com/linkedin/Liger-Kernel
    # apply_liger_kernel(config, model_args, is_trainable, require_logits=(finetuning_args.stage not in ["pt", "sft"]))

    model = None
    lazy_load = False
    if model_args.use_unsloth:
        if model_args.adapter_name_or_path is not None:
            lazy_load = True
        elif is_trainable:
            model = load_unsloth_pretrained_model(config, model_args)

    if model_args.quantization_method == QuantizationMethod.GPTQ.value:
        quantize_config = BaseQuantizeConfig(
            bits=model_args.quantization_bit,
            group_size=128,       # Group size (optional, can be None)
            desc_act=False        # Disable activation descriptor (optional)
        )
        model = AutoGPTQForCausalLM.from_pretrained(model_args.model_name_or_path, bits=model_args.quantization_bit, group_size=128, quantize_config=quantize_config)

    if model is None and not lazy_load:
        init_kwargs["config"] = config
        init_kwargs["pretrained_model_name_or_path"] = model_args.model_name_or_path
        model = AutoModelForCausalLM.from_pretrained(**init_kwargs)

    if not lazy_load:
        patch_model(model, tokenizer, model_args, is_trainable)
        register_autoclass(config, model, tokenizer)

    model = init_adapter(config, model, model_args, finetuning_args, is_trainable)

    if not is_trainable:
        model.requires_grad_(False)
        for param in model.parameters():
            if param.data.dtype == torch.float32 and model_args.compute_dtype != torch.float32:
                param.data = param.data.to(model_args.compute_dtype)

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

    return model