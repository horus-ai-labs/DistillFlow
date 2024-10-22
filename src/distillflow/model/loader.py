from typing import TypedDict, Optional, Dict, Any

from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
# from auto_gptq import AutoGPTQForCausalLM
from transformers import PreTrainedTokenizer, ProcessorMixin, AutoConfig, AutoTokenizer, AutoProcessor, PreTrainedModel, \
    PretrainedConfig, AutoModelForCausalLM, GPTQConfig

from .adapter import init_adapter
from .finetuning_args import FinetuningArguments
from .quantization import configure_quantization, QuantizationMethod
from .unsloth import load_unsloth_pretrained_model
from ..misc.logger import get_logger
from .args import ModelArguments
import torch

from ..misc import count_parameters

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
        # patch_model(model, tokenizer, model_args, is_trainable, add_valuehead)
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