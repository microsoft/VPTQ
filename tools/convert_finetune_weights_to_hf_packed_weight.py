# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import math
import torch

import os.path as osp
import time

from argparse import ArgumentParser
from LUTobq.ist.llama import llama_eval
from LUTobq.ist.data import get_loaders
from LUTobq.utils.config import Config
from safetensors import safe_open
import glob
from transformers import AutoTokenizer


def pack_index(
    indice: torch.Tensor,
    index_bits: int,
    res_indice: torch.Tensor = None,
    res_bits: int = 0,
    index_dtype: torch.dtype = torch.uint16,
    as_dtype: torch.dtype = torch.int32,
) -> torch.Tensor:
    total_bits = index_bits + res_bits
    assert total_bits <= 32, f"total index bits {total_bits} should be less than 32"
    assert as_dtype in [torch.int32], "as_dtype should be int32"

    # upcast the indice to uint64 to avoid overflow on signed bit
    if res_indice is not None:
        merged_indice = (res_indice.view(index_dtype).to(torch.uint64).view(torch.int64) << index_bits) | indice.view(
            index_dtype
        ).to(torch.uint64).view(torch.int64)
    else:
        merged_indice = indice.view(index_dtype).to(torch.uint64).view(torch.int64)

    # merge the indice
    wf = torch.arange(0, total_bits).to(merged_indice.device).view(1, 1, 1, -1)
    out = torch.bitwise_right_shift(merged_indice.unsqueeze(-1), wf)
    torch.bitwise_and(out, 1, out=out)
    out = out.reshape(*merged_indice.shape[:-1], -1)
    paded_bits = (32 - out.reshape(*merged_indice.shape[:-1], -1).shape[-1] % 32) % 32
    out = torch.nn.functional.pad(
        out,
        (0, paded_bits),
        value=0,
        mode="constant",
    ).reshape(*merged_indice.shape[:-1], -1, 32)
    wf1 = torch.arange(0, 32, 1).to(merged_indice.device).view(1, 1, 1, -1)
    out = torch.bitwise_left_shift(out, wf1)
    out = out.sum(dim=-1).to(torch.uint32).view(as_dtype)

    unpack_indices = unpack_index_tensor(
        out,
        index_bits,
        indice.shape[-1],
        res_bits,
        res_indice.shape[-1] if res_indice is not None else 0,
        index_dtype=index_dtype,
        as_dtype=as_dtype,
    )
    assert torch.allclose(
        indice.view(index_dtype).to(torch.int64),
        unpack_indices[0],
    )

    assert torch.allclose(
        indice.view(index_dtype).to(torch.int64),
        unpack_index_tensor(
            out,
            index_bits,
            indice.shape[-1],
            res_bits,
            res_indice.shape[-1] if res_indice is not None else 0,
            index_dtype=index_dtype,
            as_dtype=as_dtype,
        )[0],
    )
    if res_indice is not None:
        assert torch.allclose(
            res_indice.view(index_dtype).to(torch.int64),
            unpack_index_tensor(
                out,
                index_bits,
                indice.shape[-1],
                res_bits,
                res_indice.shape[-1] if res_indice is not None else 0,
                index_dtype=index_dtype,
                as_dtype=as_dtype,
            )[1],
        )
    return out


def unpack_index_tensor(
    pack_tensor: torch.Tensor,
    index_bits: int,
    num_elements: int,
    res_bits: int = 0,
    num_res_elements: int = 0,
    index_dtype: torch.dtype = torch.uint16,
    as_dtype: torch.dtype = torch.int32,
) -> torch.Tensor:
    total_bits = index_bits + res_bits
    wf = torch.arange(0, 32, 1).to(pack_tensor.device).view(1, 1, 1, -1)
    out = torch.bitwise_right_shift(torch.unsqueeze(pack_tensor, -1), wf)
    torch.bitwise_and(out, 1, out=out)
    pad_size = (pack_tensor.shape[-1] * 32) % (index_bits * num_elements + res_bits * num_res_elements)
    out = out.reshape(*pack_tensor.shape[:-1], -1)
    if pad_size > 0:
        out = out[..., :-pad_size]
    out = out.reshape(*pack_tensor.shape[:-1], -1, total_bits)
    wf1 = torch.arange(0, total_bits, 1).to(pack_tensor.device).view(1, 1, 1, -1)
    out = torch.bitwise_left_shift(out, wf1).sum(dim=-1)

    unpack_indice = out.to(torch.uint64).view(torch.int64)

    indices = (unpack_indice & ((1 << index_bits) - 1)).view(torch.uint64).to(torch.int64)
    indices = indices.squeeze()

    if res_bits > 0:
        res_indices = ((unpack_indice >> index_bits) & ((1 << index_bits) - 1)).view(torch.uint64).to(torch.int64)
        res_indices = res_indices.squeeze()
    else:
        res_indices = None

    return indices, res_indices


def dtype_convert(data, from_dtype, to_dtype, as_type):
    data = data.view(from_dtype).to(to_dtype).view(as_type)
    return data


def convert_idx_dtype(model, from_dtype, to_dtype, as_type):
    print(f"converting model indices from {from_dtype} " f"to {to_dtype} as {as_type}")

    quant_config = {}
    for mod_name, sub_mod in model.named_modules():
        if "qlinear" in str(type(sub_mod)):
            sub_mod.cuda()
            # print(
            #     f'---debug---'
            #     f'index shape: {sub_mod.indices.shape}, '
            #     f'dtype: {sub_mod.indices.dtype}')

            if sub_mod.indices.dtype == torch.int64:
                sub_mod.indices.data = dtype_convert(
                    sub_mod.indices.data, sub_mod.indices.data.dtype, to_dtype, as_type
                )
            else:
                sub_mod.indices.data = dtype_convert(sub_mod.indices.data, from_dtype, to_dtype, as_type)

            if hasattr(sub_mod, "res_indices") and sub_mod.res_indices is not None:
                if sub_mod.res_indices.dtype == torch.int64:
                    sub_mod.res_indices.data = dtype_convert(
                        sub_mod.res_indices.data, sub_mod.res_indices.data.dtype, to_dtype, as_type
                    )
                else:
                    sub_mod.res_indices.data = dtype_convert(sub_mod.res_indices.data, from_dtype, to_dtype, as_type)

            if hasattr(sub_mod, "outlier_indices") and sub_mod.outlier_indices is not None:
                if sub_mod.outlier_indices.dtype == torch.int64:
                    sub_mod.outlier_indices.data = dtype_convert(
                        sub_mod.outlier_indices.data, sub_mod.outlier_indices.data.dtype, to_dtype, as_type
                    )
                else:
                    sub_mod.outlier_indices.data = dtype_convert(
                        sub_mod.outlier_indices.data, from_dtype, to_dtype, as_type
                    )

            if sub_mod.perm.dtype == torch.int64:
                sub_mod.perm.data = dtype_convert(sub_mod.perm.data, sub_mod.perm.data.dtype, to_dtype, as_type)
            else:
                sub_mod.perm.data = dtype_convert(sub_mod.perm.data, from_dtype, to_dtype, as_type)

            sub_mod.indices.data = pack_index(
                indice=sub_mod.indices,
                index_bits=int(math.log2(sub_mod.num_centroids)),
                res_indice=sub_mod.res_indices,
                res_bits=int(math.log2(sub_mod.num_res_centroids)),
                index_dtype=to_dtype,
            ).data

            sub_mod.res_indices = None

            # print(f'sub_mod.indices: {sub_mod.indices.shape}')

            # assert (sub_mod.fast_dequant() - sub_mod.dequant()).max().item() < 0.001
            sub_mod.cpu()
            quant_config[mod_name] = sub_mod.init_args
            quant_config[mod_name]["is_indice_packed"] = True
            #  handle missing key indices_as_float
            if "indices_as_float" not in quant_config[mod_name]:
                quant_config[mod_name]["indices_as_float"] = False

    model.config.quant_config = quant_config
    return model


# eval ppl to check if the model is loaded correctly
def eval_ppl(qmodel, config):
    tick = time.time()
    print(f'start time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')

    # force seqlen to fast eval
    qmodel.seqlen = 2048

    # fast eval model accuracy to check if the model is loaded correctly
    datasets = ["wikitext2"]
    for dataset in datasets:
        dataloader, testloader = get_loaders(dataset, model=config.model_args.model_name, seqlen=qmodel.seqlen)
        print(dataset)
        if "llama" in config.model_args.model_name:
            ppl = llama_eval(qmodel, testloader, "cuda:0")
        elif "opt" in config.model_args.model_name:
            assert True, "opt is not supported"
        print(f"ppl_{dataset}: {ppl}")

    print(f'end time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())},' f'duration: {time.time()-tick} seconds')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base_model", type=str, default=None, help="base model to load")
    parser.add_argument("--load_model", type=str, default=None, help="load model.pt from file")
    parser.add_argument("--load_state", type=str, default=None)
    parser.add_argument("--from_type", type=str, default="uint16")
    parser.add_argument("--to_type", type=str, default="uint16")
    parser.add_argument("--as_type", type=str, default="int16")

    args = parser.parse_args()

    index_type_dict = {
        "uint16": torch.uint16,
        "int16": torch.int16,
        "uint32": torch.uint32,
        "int32": torch.int32,
        "uint64": torch.uint64,
        "int64": torch.int64,
    }

    try:
        from_type = index_type_dict[args.from_type]
    except KeyError:
        raise ValueError(f"Unsupported from index type {args.from_type}")
    try:
        to_type = index_type_dict[args.to_type]
    except KeyError:
        raise ValueError(f"Unsupported to index type {args.to_type}")
    try:
        as_type = index_type_dict[args.as_type]
    except KeyError:
        raise ValueError(f"Unsupported as index type {args.as_type}")

    print(f"load model from {args.load_model}, from_type: {from_type}, to_type: {to_type}")

    # load pt model
    args = parser.parse_args()
    experiment_path, _ = osp.split(args.load_model)
    args.load_config = osp.join(experiment_path, "config.json")
    config = Config(args.load_config)

    qmodel = torch.load(args.load_model, map_location="cpu")

    def load_model_shards(state_path):
        state_dict = {}
        shard_paths = glob.glob(osp.join(state_path, "model-*-of-*.safetensors"))
        if len(shard_paths) == 0:
            shard_paths = glob.glob(osp.join(state_path, "model.safetensors"))
        for shard_path in shard_paths:
            path = osp.join(state_path, shard_path)
            with safe_open(path, framework="pt", device="cpu") as f:
                for k in f.keys():
                    state_dict[k] = f.get_tensor(k)
        return state_dict

    # load model state from checkpoint
    if args.load_state is not None:
        print(f"load model state from {args.load_state}")
        combined_state_dict = load_model_shards(args.load_state)
        qmodel.load_state_dict(combined_state_dict)

    print(f"model seqlen {qmodel.seqlen}")

    if qmodel.seqlen != 4096 and (
        "llama2" in config.model_args.model_name.lower() or "llama-2" in config.model_args.model_name.lower()
    ):
        print("WARNING! LLama-2 model should set seqlen=4096")
    qmodel.eval()

    # print('--- eval loaded model ---')
    # eval_ppl(qmodel, config)

    model = convert_idx_dtype(qmodel, from_type, to_type, as_type)

    # model = convert_idx_dtype(qmodel, from_type, to_type)
    model.generation_config.cache_config = None
    model.generation_config.watermarking_config = None

    name = args.base_model.split("/")[-1:][0]

    print(f"saving model to {name}")
    model.save_pretrained(name)
    tokenizer = AutoTokenizer.from_pretrained(f"{args.base_model}", legacy=False)
    tokenizer.save_pretrained(name)

    import vptq

    model = vptq.AutoModelForCausalLM.from_pretrained(name, device_map="auto")
    # print(measure_token_latency(args, model.half(), tokenizer, 1))
    # print("done")
    model = model.half()
    print("--- eval saved model ---")
    eval_ppl(model, config)
