# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import gc
import unittest

import torch
import transformers

import vptq


class TestGeneration(unittest.TestCase):

    model_name = "VPTQ-community/Meta-Llama-3.1-8B-Instruct-v8-k65536-256-woft"
    test_case = "Explain: Do Not Go Gentle into That Good Night"

    max_new_tokens = 50
    pad_token_id = 2

    @classmethod
    def setUpClass(cls):
        """
        Setup quantized model
        """
        cls.tokenizer = transformers.AutoTokenizer.from_pretrained(
            cls.model_name
        )
        cls.model = vptq.AutoModelForCausalLM.from_pretrained(
            cls.model_name, device_map="auto"
        )

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

    def test_generation(self):
        # TODO(ying): Add more meaningful unit tests.
        inputs = self.tokenizer(self.test_case, return_tensors="pt").to("cuda")
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.pad_token_id,
        )
        output_string = self.tokenizer.decode(out[0], skip_special_tokens=True)

        print(output_string)


if __name__ == "__main__":
    unittest.main()
