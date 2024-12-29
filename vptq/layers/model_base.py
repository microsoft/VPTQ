# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import glob
import importlib.util
from pathlib import Path

import accelerate
import huggingface_hub
import safetensors
import torch
import transformers
from tqdm import tqdm

from vptq.layers.vqlinear import VQuantLinear


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


def make_quant_linear(module, quant_conf, target_layer=None):
    for module_name, sub_module in tqdm(
        module.named_modules(),
        total=len(list(module.named_modules())),
        desc="Replacing linear layers..."
    ):
        if module_name in quant_conf:
            layer_conf = quant_conf[module_name]

            new_module = target_layer(
                **layer_conf,
                enable_proxy_error=False,
                dtype=sub_module.weight.dtype
            )
            set_op_by_name(module, module_name, new_module)
            del sub_module
    return


def attach_execution_device_hook(
    module: torch.nn.Module,
    execution_device,
    skip_keys=None,
    preload_module_classes=None,
    tied_params_map=None,
):
    """
    A bug of accelerate, https://github.com/huggingface/accelerate/issues/3060
    we just hook it here to fix the bug.
    """
    if not hasattr(module, "_hf_hook") and len(module.state_dict()) > 0:
        accelerate.hooks.add_hook_to_module(
            module,
            accelerate.hooks.AlignDevicesHook(
                execution_device,
                skip_keys=skip_keys,
                tied_params_map=tied_params_map
            ),
        )

    # Break the recursion if we get to a preload module.
    if (
        preload_module_classes is not None and
        module.__class__.__name__ in preload_module_classes
    ):
        return

    for child in module.children():
        attach_execution_device_hook(
            child,
            execution_device,
            tied_params_map=tied_params_map,
            preload_module_classes=preload_module_classes,
        )


class AutoModelForCausalLM(transformers.AutoModelForCausalLM):

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path, *model_args, **kwargs
    ):
        init_contexts = [
            transformers.modeling_utils.no_init_weights(),
            accelerate.init_empty_weights(),
        ]
        auto_conf = transformers.AutoConfig.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        cls_kwargs = {}
        torch_dtype = kwargs.get("dtype", auto_conf.torch_dtype)
        cls_kwargs["torch_dtype"] = torch_dtype
        with transformers.utils.generic.ContextManagers(init_contexts):
            model = cls.from_config(auto_conf, *model_args, **cls_kwargs)

        target_layer = VQuantLinear
        quantization_config = auto_conf.quantization_config
        config_for_layers = quantization_config['config_for_layers']

        # replace linear layers with quantized linear layers
        with transformers.utils.generic.ContextManagers([
            accelerate.init_empty_weights()
        ]):
            make_quant_linear(
                model, config_for_layers, target_layer=target_layer
            )

        no_split_module_classes = [
            i[1].__class__.__name__
            for i in model.named_modules()
            if i[0].endswith(".0")
        ]

        device_map = kwargs.pop("device_map", None)
        max_memory = kwargs.pop("max_memory", None)

        if device_map is None:
            num_gpus = torch.cuda.device_count()
            device_names = [f"cuda:{i}" for i in range(num_gpus)]
            device_names.append("cpu")  # Include CPU for offloading

            gpu_memory = {
                device: "auto"
                for device in device_names
                if device.startswith("cuda")
            }
            cpu_memory = {"cpu": "auto"}

            max_memory = {**gpu_memory, **cpu_memory}

            # Infer device map with CPU as a fallback
            device_map = accelerate.infer_auto_device_map(
                model,
                max_memory=max_memory,
                no_split_module_classes=no_split_module_classes,
                dtype=torch_dtype,
                allow_cpu_offload=True,  # Allow offloading to CPU
            )

        if Path(pretrained_model_name_or_path).exists():
            checkpoint = pretrained_model_name_or_path
        else:  # remote
            token_arg = {"token": kwargs.get("token", None)}
            checkpoint = huggingface_hub.snapshot_download(
                repo_id=pretrained_model_name_or_path,
                ignore_patterns=["*.bin"],
                **token_arg
            )
            weight_bins = glob.glob(
                str(Path(checkpoint).absolute() / "*.safetensors")
            )
            index_json = glob.glob(
                str(Path(checkpoint).absolute() / "*.index.json")
            )
            pytorch_model_bin = glob.glob(
                str(Path(checkpoint).absolute() / "pytorch_model.bin")
            )
            if len(index_json) > 0:
                checkpoint = index_json[0]
            elif len(pytorch_model_bin) > 0:
                pass
            elif len(weight_bins) > 0:
                torch.save(
                    safetensors.torch.load_file(weight_bins[0]),
                    checkpoint + "/pytorch_model.bin"
                )

        # force to use one GPU as most as possible
        model_buffer_size = accelerate.utils.modeling.compute_module_sizes(
            model, dtype=torch_dtype
        )[""]
        local_max_memory = accelerate.utils.modeling.get_max_memory()

        if (0 in local_max_memory
           ) and (local_max_memory[0] * 0.9 > model_buffer_size):
            local_max_memory = {0: local_max_memory[0]}

        if max_memory is None:
            max_memory = local_max_memory

        accelerate.hooks.attach_execution_device_hook = \
            attach_execution_device_hook
        model = accelerate.load_checkpoint_and_dispatch(
            model,
            checkpoint=checkpoint,
            device_map=device_map,
            max_memory=max_memory,
            no_split_module_classes=no_split_module_classes[0],
            dtype=torch_dtype,
            preload_module_classes=["VQuantLinear"]
        )

        # check cuda kernel exist
        if importlib.util.find_spec("vptq.cuda_ops") is not None:
            pass
        else:
            print((
                "!!! Warning !!!: CUDA kernels are not found, "
                "please check CUDA and VPTQ installation."
            ))
            print((
                "!!! Warning !!!: Running on Torch implementations, "
                "which is extremely slow."
            ))
        model.eval()

        torch.cuda.empty_cache()
        return model
