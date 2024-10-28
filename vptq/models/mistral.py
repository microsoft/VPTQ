# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# Model from https://github.com/Cornell-RelaxML/quip-sharp/blob/main/model/llama.py

import torch


def get_mistral(model_name, seqlen=None):

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import MistralForCausalLM
    model = MistralForCausalLM.from_pretrained(
        model_name, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16
    )
    model.seqlen = seqlen
    return model
