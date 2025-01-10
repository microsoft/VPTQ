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
from sentence_transformers import SentenceTransformer
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


def make_quant_linear(module, quant_conf, name="", target_layer=None):
    for module_name, sub_module in tqdm(
        module.named_modules(), total=len(list(module.named_modules())), desc="Replacing linear layers..."
    ):
        if module_name in quant_conf:
            layer_conf = quant_conf[module_name]
            new_module = target_layer(**layer_conf, dtype=sub_module.weight.dtype)
            # print(f"Replacing {module_name} with {new_module}, {layer_conf}")
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
            accelerate.hooks.AlignDevicesHook(execution_device, skip_keys=skip_keys, tied_params_map=tied_params_map),
        )

    # Break the recursion if we get to a preload module.
    if preload_module_classes is not None and module.__class__.__name__ in preload_module_classes:
        return

    for child in module.children():
        attach_execution_device_hook(
            child,
            execution_device,
            tied_params_map=tied_params_map,
            preload_module_classes=preload_module_classes,
        )


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
            accelerate.hooks.AlignDevicesHook(execution_device, skip_keys=skip_keys, tied_params_map=tied_params_map),
        )

    # Break the recursion if we get to a preload module.
    if preload_module_classes is not None and module.__class__.__name__ in preload_module_classes:
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
    def from_pretrained(cls, pretrained_model_name_or_path, auto_conf=None, *args, **kwargs):
        init_contexts = [
            transformers.modeling_utils.no_init_weights(),
            accelerate.init_empty_weights(),
        ]
        if auto_conf is None:
            auto_conf = transformers.AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        cls_kwargs = {}
        torch_dtype = kwargs.get("dtype", auto_conf.torch_dtype)
        cls_kwargs["torch_dtype"] = torch_dtype
        with transformers.utils.generic.ContextManagers(init_contexts):
            model = cls.from_config(auto_conf, *args, **cls_kwargs)

        target_layer = VQuantLinear
        if hasattr(auto_conf, "text_config"):
            config_for_layers = auto_conf.text_config.quantization_config
        else:
            config_for_layers = auto_conf.quantization_config['config_for_layers']

        # replace linear layers with quantized linear layers
        with transformers.utils.generic.ContextManagers([accelerate.init_empty_weights()]):
            make_quant_linear(model, config_for_layers, target_layer=target_layer)

        no_split_module_classes = [i[1].__class__.__name__ for i in model.named_modules() if i[0].endswith(".0")]

        device_map = kwargs.pop("device_map", None)
        max_memory = kwargs.pop("max_memory", None)

        if device_map is None:
            num_gpus = torch.cuda.device_count()
            device_names = [f"cuda:{i}" for i in range(num_gpus)]
            device_names.append("cpu")  # Include CPU for offloading

            gpu_memory = {device: "auto" for device in device_names if device.startswith("cuda")}
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
            token_arg = {"token": kwargs.get("token")}
            checkpoint = huggingface_hub.snapshot_download(
                repo_id=pretrained_model_name_or_path, ignore_patterns=["*.bin"], **token_arg
            )
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

        if 0 in local_max_memory and local_max_memory[0] * 0.9 > model_buffer_size:
            local_max_memory = {0: local_max_memory[0]}

        if max_memory is None:
            max_memory = local_max_memory

        accelerate.hooks.attach_execution_device_hook = attach_execution_device_hook
        model = accelerate.load_checkpoint_and_dispatch(
            model,
            checkpoint=checkpoint,
            device_map=device_map,
            max_memory=max_memory,
            no_split_module_classes=no_split_module_classes[0],
            dtype=torch_dtype,
            preload_module_classes=["VQuantLinear"],
        )

        # check cuda kernel exist
        if importlib.util.find_spec("vptq.ops") is not None:
            pass
        else:
            print("!!! Warning !!!: CUDA kernel not found, please check CUDA and VPTQ installation.")
            print("!!! Warning !!!: Running on Torch Implementation, which is extremely slow.")

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


class AutoModelForSentenceEmbeddings(SentenceTransformer):

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
        }
        model = SentenceTransformer(
            pretrained_model_name_or_path,
            model_kwargs=model_kwargs,
            trust_remote_code=True,
        )
        print(model._modules["0"])
        text_config = model._modules["0"]._modules["auto_model"].config
        model._modules["0"]._modules["auto_model"].embedding_model = None
        torch.cuda.empty_cache()

        model._modules["0"]._modules["auto_model"] = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path, auto_conf=text_config, *args, **kwargs
        )

        model.eval()
        torch.cuda.empty_cache()
        return model
