# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# Model from https://github.com/Cornell-RelaxML/quip-sharp/blob/main/model/llama.py
# Evluation from https://github.com/IST-DASLab/gptq/blob/main/eval.py

import os
import time

import torch
import torch.multiprocessing as mp
import torch.nn as nn
from tqdm import tqdm

from vptq.layers.vqlinear import VQuantLinear
from vptq.quantize_executer import quantize_executer
from vptq.utils.layer_utils import find_layers, replace_layer


# get llama model
def get_llama(model_name, seqlen=None):

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(
        model_name, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16
    )

    if seqlen is not None:
        model.seqlen = seqlen
    return model


# quant llama model
def quant_llama(model, args, quant_args, dev='cuda'):
    # model.model.required_grad = False
    print('Starting VPTQ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    # get decoder layers
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    if hasattr(model.model, 'rotary_emb'):
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    print(f'model dtype: {dtype}')

    # save model to cpu
    model = model.cpu()

    layers[0] = layers[0].cpu()

    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    model.config.use_cache = use_cache

    torch.cuda.empty_cache()

    model.model.embed_tokens = model.model.embed_tokens.to('cpu')
    # fix for llama-3.1
    if hasattr(model.model, 'rotary_emb'):
        model.model.rotary_emb = model.model.rotary_emb.to('cpu')
    model.model.norm = model.model.norm.to('cpu')
    layers[0] = layers[0].to('cpu')
    model = model.cpu()

    # multiple gpus VPTQ
    quantizers = {}
    layers = model.model.layers

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
        print(f'gpu {gpu_idx} tasks: {task}')

    # init multiprocessing
    processes = []
    mq_manager = mp.get_context('spawn').Manager()
    input_queues = mq_manager.Queue()
    output_queues = mq_manager.Queue()

    if args.num_gpus == 1:
        layer_state_dicts, layer_qlinear_args = quantize_executer(0, tasks[0], args, quant_args, None, None)
    else:
        for gpu_idx in range(args.num_gpus):
            # we have to set CUDA_VISIBLE_DEVICES here
            # cuml only supports to run on GPU:0
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
                )
            )

            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    print(f'----quantization done ...---- {time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}')

    # init quantized model
    model_name = model.model.config._name_or_path
    # qmodel = init_quantized_llama(model_name, args, quant_args).cpu()

    if args.num_gpus > 1:
        layer_state_dicts = {}
        layer_qlinear_args = {}

        # load save qlinear from files to avoid memory overflow
        if args.save_qlinear:
            for layer_idx in range(len(layers)):
                # load to cpu
                layer_state_dicts[layer_idx] = torch.load(
                    f'{args.output_dir}/qlinear_layer_state_{layer_idx}.pt', map_location='cpu',
                    weights_only=False
                )
                # bypass KeyError: torch.uint16
                for key, value in layer_state_dicts[layer_idx].items():
                    if "indices" in key:
                        layer_state_dicts[layer_idx][key] = value.view(torch.uint16)
                layer_qlinear_args[layer_idx] = torch.load(
                    f'{args.output_dir}/qlinear_args_{layer_idx}.pt', map_location='cpu',
                    weights_only=False
                )
        else:
            while not output_queues.empty():
                (gpu_id, layer_idx, _layer_state_dict, _layer_qlinear_args) = output_queues.get()
                layer_state_dicts[layer_idx] = _layer_state_dict
                layer_qlinear_args[layer_idx] = _layer_qlinear_args
                print(f'gpu {gpu_id} layer {layer_idx} quantized')

    # check if all layers are quantized
    if len(layer_state_dicts) != len(layers):
        print('Error: not all layers are quantized')
        exit(1)

    qmodel = get_quantized_llama(model_name, args.seq_len, layer_state_dicts, layer_qlinear_args)

    model = qmodel

    print(f'qmodel: {model}')

    torch.cuda.empty_cache()
    return model, quantizers


def get_quantized_llama(model_name, seqlen, layer_state_dicts, layer_qlinear_args):

    # print(f'layer_state_dicts: {layer_state_dicts.keys()}')
    model = get_llama(model_name, seqlen=seqlen)
    dtype = next(iter(model.parameters())).dtype
    layers = model.model.layers

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
            module_name = name.split('.')[-1]
            replace_layer(layer, module_name, qlayer)

        # convert dtype
        # print(f'default dtype: {dtype}')
        for param_name, param in layer_state_dict.items():
            if layer_state_dict[param_name].dtype not in [
                dtype, torch.int64, torch.int32, torch.int16, torch.int8, torch.uint64, torch.uint32, torch.uint16,
                torch.uint8, torch.bool
            ]:
                layer_state_dict[param_name] = layer_state_dict[param_name].to(dtype)

        layers[layer_idx].load_state_dict(layer_state_dict)

    return model


@torch.no_grad()
def eval_llama(model, testenc, dev):
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    print(f'----Evaluating llama ...---- {current_time}')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    # fix for llama-3.1
    if hasattr(model.model, 'rotary_emb'):
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            if hasattr(model.model, 'rotary_emb'):
                cache['rotary_emb'] = model.model.rotary_emb(x=inp, position_ids=kwargs['position_ids'])
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(input_ids=batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    # get position embeddings from the model's rotary embeddings
    # for the latest huggingface transformers
    position_embeddings = model.model.rotary_emb(outs, position_ids)

    for i in tqdm(range(len(layers))):
        layer = layers[i].to(dev)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0).to(dev), 
                          attention_mask=attention_mask, 
                          position_ids=position_ids,
                          position_embeddings=position_embeddings)[0]

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache
    return ppl.item()
