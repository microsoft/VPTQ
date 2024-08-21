# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import argparse
from .ist.model_base import AutoModelForCausalLM as VQAutoModelQuantization
import transformers


def define_basic_args():
    parser = argparse.ArgumentParser(description="""run a quantized model.
A typical usage is:
    python -m vptq --model  Llama-2-7b-1.5bit --prompt "Hello, my dog is cute"
 """, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model', type=str, required=True,
                        help='float/float16 model to load, such as [mosaicml/mpt-7b]')
    parser.add_argument('--tokenizer', type=str, default="", help='default same as [model]')
    parser.add_argument('--prompt', type=str, default="once upon a time, ", help='prompt to start generation')
    #parser.add_argument('--chat',action='store_true', help='chat with the model')
    return parser


def get_valid_args(parser):
    args = parser.parse_args()
    return args

def main():
    parser = define_basic_args()
    args = get_valid_args(parser)
    print(args)

    model = VQAutoModelQuantization.from_pretrained(args.model)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer or args.model)

    inputs = tokenizer(args.prompt, return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=100, pad_token_id=2)
    print(tokenizer.decode(out[0], skip_special_tokens=True))

main()
