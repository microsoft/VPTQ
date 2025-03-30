# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import gc

import pytest
import torch
import transformers

import vptq


@pytest.fixture(scope="class")
def model_and_tokenizer():
    """Setup quantized model and tokenizer."""
    model_name = "VPTQ-community/Meta-Llama-3.1-8B-Instruct-v8-k65536-256-woft"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = vptq.AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto"
    )
    yield model, tokenizer

    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()


def test_generation(model_and_tokenizer):
    """Test text generation with the quantized model."""
    model, tokenizer = model_and_tokenizer
    test_case = "Explain: Do Not Go Gentle into That Good Night"
    max_new_tokens = 50
    pad_token_id = 2

    inputs = tokenizer(test_case, return_tensors="pt").to("cuda")
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=pad_token_id,
    )
    output_string = tokenizer.decode(out[0], skip_special_tokens=True)

    print(output_string)
    # TODO(ying): Add more meaningful assertions
