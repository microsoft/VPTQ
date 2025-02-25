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
    "-m",
    "--input_path",
    type=str,
    default="Meta-Llama-3.1-8B-Instruct-v12-k65536-4096-woft/",
)
parser.add_argument(
    "-o",
    "--output_path",
    type=str,
    default="./Meta-Llama-3.1-8B-Instruct-v12-k65536-4096-woft_processed",
)
parser.add_argument(
    "--gpu_utilization",
    type=float,
    default=None,
    help="GPU utilization for model loading",
)

if __name__ == "__main__":
    args = parser.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.input_path)
    model = vptq.AutoModelForCausalLM.from_pretrained(
        args.input_path, gpu_utilization=args.gpu_utilization, device_map="auto"
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
    import accelerate

    old_get_state_dict_from_offload = (
        transformers.modeling_utils.get_state_dict_from_offload
    )
    def new_get_state_dict_from_offload(
        module,
        module_name: str,
        state_dict,
        device_to_put_offload = "cpu",
    ):
        root = module_name[: module_name.rfind(".")]  # module name without .weight or .bias
        if not accelerate.utils.has_offloaded_params(module):
            device_to_put_offload = None
        with accelerate.utils.align_module_device(module, device_to_put_offload):
            for m_key, params in module.state_dict().items():
                if (root + f".{m_key}") in state_dict and isinstance(
                    state_dict[root + f".{m_key}"], str
                ):
                    state_dict[root + f".{m_key}"] = params

        return state_dict

    transformers.modeling_utils.get_state_dict_from_offload = (
        new_get_state_dict_from_offload
    )
    model.save_pretrained(output_path)
    transformers.modeling_utils.get_state_dict_from_offload = (
        old_get_state_dict_from_offload
    )
    tokenizer.save_pretrained(output_path)
    logger.info(f"Model and tokenizer saved to {output_path}")  # noqa: G004
