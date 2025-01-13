# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# Model from https://github.com/Cornell-RelaxML/quip-sharp/blob/main/model/llama.py

import os
import time

import torch
import torch.multiprocessing as mp
from sentence_transformers import SentenceTransformer

from vptq.layers.vqlinear import VQuantLinear
from vptq.quantize_executer import quantize_executer
from vptq.utils.layer_utils import find_layers, replace_layer


def get_nvembed(model_name, seqlen=None):

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    model_kwargs = {
        "torch_dtype": torch.bfloat16,
    }
    model = SentenceTransformer(
        model_name,
        model_kwargs=model_kwargs,
        trust_remote_code=True,
    )

    model.max_seq_length = seqlen
    return model


def quant_nvembed(model: SentenceTransformer, args, quant_args, dev="cuda"):
    # model.model.required_grad = False
    print("Starting VPTQ...")

    model = model._modules["0"]._modules["auto_model"]

    use_cache = model.config.text_config.use_cache
    model.config.text_config.use_cache = False
    # get decoder layers
    layers = model.embedding_model.layers

    model.embedding_model.embed_tokens = model.embedding_model.embed_tokens.to(dev)
    if hasattr(model.embedding_model, "rotary_emb"):
        model.embedding_model.rotary_emb = model.embedding_model.rotary_emb.to(dev)
    model.embedding_model.norm = model.embedding_model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    print(f"model dtype: {dtype}")

    # save model to cpu
    model = model.cpu()

    layers[0] = layers[0].cpu()

    model.embedding_model.embed_tokens = model.embedding_model.embed_tokens.cpu()
    model.embedding_model.norm = model.embedding_model.norm.cpu()
    model.config.text_config.use_cache = use_cache

    torch.cuda.empty_cache()

    model.embedding_model.embed_tokens = model.embedding_model.embed_tokens.to("cpu")
    # fix for llama-3.1
    if hasattr(model.embedding_model, "rotary_emb"):
        model.embedding_model.rotary_emb = model.embedding_model.rotary_emb.to("cpu")
    model.embedding_model.norm = model.embedding_model.norm.to("cpu")
    layers[0] = layers[0].to("cpu")
    model = model.cpu()

    # multiple gpus VPTQ
    quantizers = {}
    layers = model.embedding_model.layers

    print(f'----quantization start ...---- {time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}')

    # calculate task allocation
    total_layers = len(layers)
    num_gpus = min(args.num_gpus, total_layers)

    base_layers_per_gpu = total_layers // num_gpus
    remaining_layers = total_layers % num_gpus

    tasks = []
    current_layer_idx = 0

    # Distribute tasks to GPUs
    for gpu_idx in range(num_gpus):
        current_gpu_tasks = []

        # Calculate how many layers this GPU should handle
        layers_for_this_gpu = base_layers_per_gpu
        if gpu_idx < remaining_layers:
            layers_for_this_gpu += 1

        # Assign layers to this GPU
        for _ in range(layers_for_this_gpu):
            current_gpu_tasks.append((current_layer_idx, layers[current_layer_idx]))
            current_layer_idx += 1

        tasks.append(current_gpu_tasks)

    # print task allocation
    for gpu_idx in range(len(tasks)):
        task = [layer_idx for layer_idx, _ in tasks[gpu_idx]]
        print(f"gpu {gpu_idx} tasks: {task}")

    # init multiprocessing
    processes = []
    mq_manager = mp.get_context("spawn").Manager()
    input_queues = mq_manager.Queue()
    output_queues = mq_manager.Queue()

    if args.num_gpus == 1:
        layer_state_dicts, layer_qlinear_args = quantize_executer(0, tasks[0], args, quant_args, None, None)
    else:
        for gpu_idx in range(args.num_gpus):
            # we have to set CUDA_VISIBLE_DEVICES here
            # oscuml only supports to run on GPU:0
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
            p = mp.Process(
                target=quantize_executer,
                args=(
                    gpu_idx,
                    tasks[gpu_idx],
                    args,
                    quant_args,
                    input_queues,
                    output_queues,
                ),
            )

            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    print(f'----quantization done ...---- {time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}')

    # init quantized model
    model_name = model.embedding_model.config._name_or_path
    # qmodel = init_quantized_llama(model_name, args, quant_args).cpu()

    if True:
        layer_state_dicts = {}
        layer_qlinear_args = {}

        # load save qlinear from files to avoid memory overflow
        if args.save_qlinear:
            for layer_idx in range(len(layers)):
                # load to cpu
                layer_state_dicts[layer_idx] = torch.load(
                    f"{args.output_dir}/qlinear_layer_state_{layer_idx}.pt", map_location="cpu",
                    weights_only=True
                )
                # bypass KeyError: torch.uint16
                for key, value in layer_state_dicts[layer_idx].items():
                    if "indices" in key:
                        layer_state_dicts[layer_idx][key] = value.view(torch.uint16)
                layer_qlinear_args[layer_idx] = torch.load(
                    f"{args.output_dir}/qlinear_args_{layer_idx}.pt", map_location="cpu",
                    weights_only=True
                )
        else:
            while not output_queues.empty():
                (gpu_id, layer_idx, _layer_state_dict, _layer_qlinear_args) = output_queues.get()
                layer_state_dicts[layer_idx] = _layer_state_dict
                layer_qlinear_args[layer_idx] = _layer_qlinear_args
                print(f"gpu {gpu_id} layer {layer_idx} quantized")

    # check if all layers are quantized
    if len(layer_state_dicts) != len(layers):
        print("Error: not all layers are quantized")
        exit(1)

    qmodel = get_quantized_nvembed(model_name, args.seq_len, layer_state_dicts, layer_qlinear_args)

    model = qmodel

    print(f"qmodel: {model}")

    torch.cuda.empty_cache()
    return model, quantizers


def get_quantized_nvembed(model_name, seqlen, layer_state_dicts, layer_qlinear_args):
    # print(f'layer_state_dicts: {layer_state_dicts.keys()}')
    st_model = get_nvembed(model_name, seqlen=seqlen)
    model = st_model._modules["0"]._modules["auto_model"]

    dtype = next(iter(model.parameters())).dtype
    layers: list[torch.nn.Module] = model.embedding_model.layers

    for layer_idx, layer_state_dict in layer_state_dicts.items():
        # print(f'load quantized layer {layer_idx}')
        # print(f'layer_state_dict: {layer_state_dict.keys()}')
        layer = layers[layer_idx]
        ops = find_layers(layer)

        for name, op in ops.items():
            # init qlinear
            qlayer = VQuantLinear(
                **layer_qlinear_args[layer_idx][name],
                dtype=dtype,
            )
            module_name = name.split(".")[-1]
            replace_layer(layer, module_name, qlayer)

        # convert dtype
        # print(f'default dtype: {dtype}')
        for param_name, param in layer_state_dict.items():
            if layer_state_dict[param_name].dtype not in [
                dtype,
                torch.int64,
                torch.int32,
                torch.int16,
                torch.int8,
                torch.uint64,
                torch.uint32,
                torch.uint16,
                torch.uint8,
                torch.bool,
            ]:
                layer_state_dict[param_name] = layer_state_dict[param_name].to(dtype)

        layers[layer_idx].load_state_dict(layer_state_dict)

    return st_model
