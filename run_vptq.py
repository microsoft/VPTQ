# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import json
import os.path as osp
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch.multiprocessing import set_start_method
from transformers import AutoTokenizer, HfArgumentParser, set_seed

from vptq.models.llama import eval_llama, get_llama, quant_llama
from vptq.models.mistral import get_mistral
from vptq.models.qwen import eval_qwen, get_qwen, quant_qwen
from vptq.quantizer import QuantizationArguments
from vptq.utils.data import get_data_loader
from vptq.utils.pack import pack_model


@dataclass
class VPTQArguments:
    model_name: str = field(default='meta-llama/Llama-2-7b-hf')
    seq_len: Optional[int] = field(default=None)
    quant_step: int = field(default=1)
    percdamp: float = field(default=0.01)
    blocksize: int = field(default=128)
    output_dir: str = field(default='outputs')
    seed: int = field(default=0)
    eval: bool = field(default=False)
    new_eval: bool = field(default=False)
    save_model: bool = field(default=False)
    save_packed_model: bool = field(default=False)
    disable_actorder: bool = field(default=False)
    hessian_path: Optional[str] = field(default=None)
    inv_hessian_path: Optional[str] = field(default=None)
    num_gpus: int = field(default=1)
    eval_nsamples: int = field(default=128)
    save_qlinear: bool = field(default=False)


if __name__ == "__main__":
    parser = HfArgumentParser((VPTQArguments, QuantizationArguments))
    args, quant_args = parser.parse_args_into_dataclasses()

    # set output folder based on time
    args.output_dir = osp.join(args.output_dir, time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()))

    set_start_method('spawn')

    set_seed(args.seed)

    if 'llama' in args.model_name.lower():
        model = get_llama(args.model_name)
    elif 'qwen' in args.model_name.lower():
        model = get_qwen(args.model_name)
    elif 'mistral' in args.model_name.lower():
        model = get_mistral(args.model_name)
    else:
        raise ValueError(f'Unsupported model: {args.model_name}')

    # set sequence length
    if args.seq_len or model.seqlen is None:
        model.seqlen = args.seq_len
    print(f'model sequence length: {model.seqlen}')

    model.eval()

    tick = time.time()
    print(f'exp time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
    print(f'args: {args}')
    print(f'quant_args: {quant_args}')

    if 'llama' in args.model_name.lower() or \
            'mistral' in args.model_name.lower():
        model, quantizers = quant_llama(model, args, quant_args)
    elif 'qwen' in args.model_name.lower():
        model, quantizers = quant_qwen(model, args, quant_args)
    else:
        raise ValueError(f'Unsupported model: {args.model_name}')

    # save model, not for inference
    if args.save_model:
        model_path = osp.join(args.output_dir, 'model/')
        model.save_pretrained(model_path)
        print(f'save model to {model_path}')
        tokenizer = AutoTokenizer.from_pretrained(f'{args.model_name}', legacy=False)
        tokenizer.save_pretrained(model_path)
        print(f'save tokenizer to {model_path}')

    # save packed model for inference
    if args.save_packed_model:
        model_path = osp.join(args.output_dir, 'packed_model/')
        model = pack_model(model, from_type=torch.uint16, to_type=torch.uint16, as_type=torch.int16)
        model.save_pretrained(model_path, safe_serialization=False)
        print(f'save packed model to {model_path}')
        tokenizer = AutoTokenizer.from_pretrained(f'{args.model_name}', legacy=False)

        tokenizer.save_pretrained(model_path)
        print(f'save tokenizer to {model_path}')

        # reload packed model to evaluate
        import vptq
        model = vptq.AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
        print(f'load packed model from {model_path}')

    model.eval()
    if args.eval:
        datasets = ['wikitext2', 'c4']
    if args.new_eval:
        datasets = ['wikitext2', 'c4-new']

    seqlens = [2048, 4096, 8192]

    # store results
    results = {}

    for seqlen in seqlens:
        model.seqlen = seqlen
        for dataset in datasets:
            dataloader, testloader = get_data_loader(dataset,
                                                     seed=args.seed,
                                                     model=args.model_name,
                                                     seqlen=model.seqlen)
            print(dataset)
            if 'llama' in args.model_name.lower() \
                    or 'mistral' in args.model_name.lower():
                ppl = eval_llama(model, testloader, 'cuda')
            elif 'qwen' in args.model_name.lower():
                ppl = eval_qwen(model, testloader, 'cuda')
            else:
                raise ValueError(f'Unsupported model: {args.model_name}')

            if f'ctx_{seqlen}' not in results:
                results[f'ctx_{seqlen}'] = {}
            results[f'ctx_{seqlen}'][dataset] = ppl

        with open(osp.join(args.output_dir, 'ppl_results.json'), "w") as f:
            json.dump(results, f, indent=2)
