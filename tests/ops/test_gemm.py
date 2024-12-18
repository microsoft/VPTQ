# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from typing import Tuple

import torch
import transformers

import vptq

model_name = "VPTQ-community/Meta-Llama-3.1-70B-Instruct-v8-k65536-0-woft"


def infer():
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    m = vptq.AutoModelForCausalLM.from_pretrained(
        model_name, device_map='auto')

    inputs = tokenizer(
        "Explain: Do Not Go Gentle into That Good Night", return_tensors="pt"
    ).to("cuda")
    out = m.generate(**inputs, max_new_tokens=100, pad_token_id=2)
    print(tokenizer.decode(out[0], skip_special_tokens=True))


def load_parameters() -> Tuple[torch.Tensor]:
    m = vptq.AutoModelForCausalLM.from_pretrained(
        model_name, device_map='auto')
    quantized_params = m.model.state_dict()

    centriods = quantized_params["layers.0.mlp.up_proj.centroids.weight"]
    indices = quantized_params["layers.0.mlp.up_proj.indices"]
    perm = quantized_params["layers.0.mlp.down_proj.perm"]
    weight_scale = quantized_params["layers.0.mlp.down_proj.weight_scale"]
    weight_bias = quantized_params["layers.0.mlp.down_proj.weight_bias"]

    return centriods, indices, perm, weight_scale, weight_bias


infer()
