# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import glob
from pathlib import Path

import accelerate
import huggingface_hub
import safetensors
import torch
import transformers
from tqdm import tqdm

from .vqlinear import VQuantLinear


def set_op_by_name(layer, name, new_module):
    levels = name.split(".")
    if len(levels) > 1:
        mod_ = layer
        for l_idx in range(len(levels) - 1):
            if levels[l_idx].isdigit():  # noqa:SIM108
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_module)
    else:
        setattr(layer, name, new_module)


def make_quant_linear(module, quant_conf, name="", target_layer=None):
    for module_name, sub_module in tqdm(module.named_modules(),
                                        total=len(list(module.named_modules())),
                                        desc="Replacing linear layers..."):
        if module_name in quant_conf:
            layer_conf = quant_conf[module_name]
            new_module = target_layer(**layer_conf, enable_proxy_error=False, dtype=sub_module.weight.dtype)
            # print(f"Replacing {module_name} with {new_module}, {layer_conf}")
            set_op_by_name(module, module_name, new_module)
            del sub_module
    return


class AutoModelForCausalLM(transformers.AutoModelForCausalLM):

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        init_contexts = [
            transformers.modeling_utils.no_init_weights(),
            accelerate.init_empty_weights(),
        ]
        auto_conf = transformers.AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        cls_kwargs = {}
        torch_dtype = kwargs.get("dtype", auto_conf.torch_dtype)
        cls_kwargs["torch_dtype"] = torch_dtype
        with transformers.utils.generic.ContextManagers(init_contexts):
            model = cls.from_config(auto_conf, *model_args, **cls_kwargs)

        target_layer = VQuantLinear
        quant_config = auto_conf.quant_config

        # replace linear layers with quantized linear layers
        with transformers.utils.generic.ContextManagers([accelerate.init_empty_weights()]):
            make_quant_linear(model, quant_config, target_layer=target_layer)

        no_split_module_classes = [i[1].__class__.__name__ for i in model.named_modules() if i[0].endswith(".0")]

        device_map = kwargs.pop("device_map", None)
        max_memory = kwargs.pop("max_memory", None)

        # device_map = accelerate.infer_auto_device_map(model, no_split_module_classes=no_split_module_classes[0],
        # dtype=torch_dtype)
        if Path(pretrained_model_name_or_path).exists():
            checkpoint = pretrained_model_name_or_path
        else:  # remote
            token_arg = {"token": kwargs.get("token", None)}
            checkpoint = huggingface_hub.snapshot_download(repo_id=pretrained_model_name_or_path,
                                                           ignore_patterns=["*.bin"],
                                                           **token_arg)
            weight_bins = glob.glob(str(Path(checkpoint).absolute() / "*.safetensors"))
            index_json = glob.glob(str(Path(checkpoint).absolute() / "*.index.json"))
            pytorch_model_bin = glob.glob(str(Path(checkpoint).absolute() / "pytorch_model.bin"))
            if len(index_json) > 0:
                checkpoint = index_json[0]
            elif len(pytorch_model_bin) > 0:
                pass
            elif len(weight_bins) > 0:
                torch.save(safetensors.torch.load_file(weight_bins[0]), checkpoint + "/pytorch_model.bin")

        # force to use one GPU as most as possible
        model_buffer_size = accelerate.utils.modeling.compute_module_sizes(model, dtype=torch_dtype)[""]
        local_max_memory = accelerate.utils.modeling.get_max_memory()
        if 0 in local_max_memory and local_max_memory[0] * 0.85 > model_buffer_size:
            local_max_memory = {0: local_max_memory[0]}
        if max_memory is None:
            max_memory = local_max_memory

        model = accelerate.load_checkpoint_and_dispatch(
            model,
            checkpoint=checkpoint,
            device_map=device_map,
            max_memory=max_memory,
            no_split_module_classes=no_split_module_classes[0],
            dtype=torch_dtype,
            # preload_module_classes=["VQuantLinear"]
        )
        
        # check cuda kernel exist
        try:
            from vptq import ops
        except ImportError:
            print('!!! Warning !!!: CUDA kernel not found, please check CUDA and VPTQ installation.')
            print('!!! Warning !!!: Running on Torch Implementation, which is extremely slow.')

        # weight_bins = glob.glob(str(Path(pretrained_model_name_or_path).absolute() / '*.safetensors'))
        # all_missing_keys = []
        # all_unexpected_keys = []
        # if len(weight_bins) > 0:
        # weights = {}
        # for i in tqdm(range(len(weight_bins)), desc="loading weights from safetensors"):
        # weights.update(safetensors.torch.load_file(weight_bins[i], device="cpu"))
        # ret = model.load_state_dict(weights, strict=False)
        # all_missing_keys.extend(ret.missing_keys)
        # all_unexpected_keys.extend(ret.unexpected_keys)
        # else:
        # assert False, "safetensors not found"
        # logger = transformers.logging.get_logger(__name__)
        # if len(all_missing_keys) > 0 or len(all_unexpected_keys) > 0:
        # logger.warning(f"Missing keys: {all_missing_keys[:5]}..., Unexpected keys: {all_unexpected_keys[:5]}...")
        model.eval()

        torch.cuda.empty_cache()
        return model
