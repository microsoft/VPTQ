# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# ruff: noqa: G004

import argparse
import logging
import os

import transformers

import vptq
from vptq.utils.pack import absorb_perm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger("vptq")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_path",
    type=str,
    default="Meta-Llama-3.1-8B-Instruct-v12-k65536-4096-woft/",
)
parser.add_argument(
    "--output_path",
    type=str,
    default="./Meta-Llama-3.1-8B-Instruct-v12-k65536-4096-woft_processed",
)

if __name__ == "__main__":
    args = parser.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.input_path)
    model = vptq.AutoModelForCausalLM.from_pretrained(
        args.input_path, device_map="auto"
    )
    #preprocessor pipeline
    # process 1-->processor 2-->processor 3-->processor 4-->processor 5
    model = absorb_perm(model)
    #other processors follows here
    # ...

    # Save the processed model
    output_path = args.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    logger.info(f"Model and tokenizer saved to {output_path}")  # noqa: G004
