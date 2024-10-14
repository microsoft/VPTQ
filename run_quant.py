# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import json
import time
from typing import Optional

import torch
from torch.multiprocessing import set_start_method
from transformers import HfArgumentParser, set_seed

from vptq.models.llama import eval_llama, get_llama
from vptq.models.mistral import eval_mistral, get_mistral
from vptq.models.qwen import eval_qwen, get_qwen
from vptq.utils.data import get_data_loader


@dataclass
class ModelArguments:
    model_name: str = field(default='meta-llama/Llama-2-7b-hf')
    dataset: str = field(default=None, metadata={"choices": [
                         'wikitext2', 'ptb', 'c4', 'pile', 'rp']})
    nsamples: int = field(default=128)
    seq_len: Optional[int] = field(default=None)
    quant_step: int = field(default=1)
    percdamp: float = field(default=0.01)
    blocksize: int = field(default=128)
    output_dir: str = field(default=None)
    seed: int = field(default=0)
    eval: bool = field(default=False)
    new_eval: bool = field(default=False)
    save_model: bool = field(default=False)
    disable_actorder: bool = field(default=False)
    save_by_layer: bool = field(default=False)
    calibration_on_cpu: bool = field(default=False)
    hessian_path: Optional[str] = field(default=None)
    inv_hessian_path: Optional[str] = field(default=None)
    num_gpus: int = field(default=1)
    eval_quant: bool = field(default=False)

if __name__ == "__main__":
    parser = HfArgumentParser(
        (QuantizationArguments, ModelArguments))

    quant_args, transpose_args, args = parser.parse_args_into_dataclasses()


    # set experiment folder
    args.output_dir = osp.join(args.output_dir, time.strftime(
        '%Y-%m-%d-%H-%M-%S', time.localtime()))

    # set tensorboard writer
    writer = get_logger(args, quant_args)

    set_start_method('spawn')

    set_seed(args.seed)

    if 'llama' in args.model_name:
        model = get_llama(args.model_name)
        # print(model)
    elif 'qwen' in args.model_name.lower():
        model = get_qwen(args.model_name)
    elif 'mistral' in args.model_name:
        model = get_mistral(args.model_name)
    
    # set sequence length
    if args.seq_len:
        model.seqlen = args.seq_len
    print(f'model sequence length: {model.seqlen})

    model.eval()

	# get data loaders
    dataloader, testloader = get_loaders(
        args.dataset, args.nsamples, args.seed, model.seqlen, args.model_name)
    
    tick = time.time()

    print(f'exp time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
    print(f'args: {args}')
    print(f'quant_args: {quant_args}')
    print(f'transpose_args: {transpose_args}')

    if 'llama' in args.model_name:
        model, quantizers = llama_sequential_ft(
            model, dataloader, args, quant_args, transpose_args, writer)
    elif 'mistral' in args.model_name:
        model, quantizers = llama_sequential_ft(
            model, dataloader, args, quant_args, transpose_args, writer)
    elif 'qwen' in args.model_name.lower():
        model, quantizers = qwen_sequential_ft(
            model, dataloader, args, quant_args, transpose_args, writer)

    writer.add_scalar('quant/exec_time(min)', (time.time()-tick)/60, 0)

    # FIX IT: save model to safe tensors
    if args.save_model:
        torch.save(model, osp.join(
            args.output_dir, 'qmodel.pt'))
    

    if args.eval:
        datasets = ['wikitext2', 'ptb', 'c4']
    if args.new_eval:
        datasets = ['wikitext2', 'c4-new']
    for dataset in datasets:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model_name, seqlen=model.seqlen)
        print(dataset)
        if 'llama' in args.model_name:
            ppl = llama_eval(model, testloader, 'cuda')
        elif 'qwen' in args.model_name.lower():
            ppl = qwen_eval(model, testloader, 'cuda')

    # save to json
    results = {'ctx_2048': {}, 'ctx_4096': {}, 'ctx_8192': {}}
    model.eval()
    if 'llama-3' or 'llama3' in args.model_name.lower():
        seqlens = [8192, 4096, 2048]
    else:
        seqlens = [4096, 2048]
    
    for seqlen in seqlens:
        model.seqlen = seqlen
        # datasets = ['wikitext2']
        datasets = ['wikitext2', 'c4-new', 'c4']
        for dataset in datasets:
            dataloader, testloader = get_loaders(dataset, seed=args.seed, model=args.model_name, seqlen=model.seqlen)
            print(dataset)
            if 'llama' in args.model_name:
                ppl = llama_eval(model, testloader, 'cuda')
            elif 'qwen' in args.model_name.lower():
                ppl = qwen_eval(model, testloader, 'cuda')
            results[f'ctx_{seqlen}'][dataset] = ppl

        with open(osp.join(args.output_dir, 'LM_results_2k_4k.json'), "w") as f:
            json.dump(results, f, indent=2)


