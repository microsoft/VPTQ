# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import argparse
import os

import transformers

from .layers.model_base import AutoModelForCausalLM as VQAutoModelQuantization


def define_basic_args():
    parser = argparse.ArgumentParser(
        description="""run a quantized model.
A typical usage is:
    python -m vptq --model  Llama-2-7b-1.5bit --prompt "Hello, my dog is cute"
 """,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--model",
                        type=str,
                        required=True,
                        help="float/float16 model to load, such as [mosaicml/mpt-7b]")
    parser.add_argument("--tokenizer", type=str, default="", help="default same as [model]")
    parser.add_argument("--prompt", type=str, default="once upon a time, there ", help="prompt to start generation")
    parser.add_argument("--chat", action="store_true", help="chat with the model")
    return parser


def eval_prompt(model, tokenizer, args):
    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    streamer = transformers.TextStreamer(tokenizer)
    model.generate(**inputs, streamer=streamer, max_new_tokens=100, pad_token_id=2)


def chat_loop(model, tokenizer, args):
    if not args.chat:
        eval_prompt(model, tokenizer, args)
        return

    if getattr(tokenizer, "chat_template", None) is None:
        print("warning: this tokenizer didn't provide chat_template.!!!")
        eval_prompt(model, tokenizer, args)
        return

    print("============================chat with the model============================")
    print("Press 'exit' to quit")
    messages = [{"role": "system", "content": "you are a math teacher."}]

    while True:
        text = input("You: ")
        if text == "exit":
            break
        messages.append({"role": "user", "content": text})
        encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
        model_inputs = encodeds.to(model.device)
        streamer = transformers.TextStreamer(tokenizer, skip_prompt=True, decode_kwargs={"skip_special_tokens": True})
        generated_ids = model.generate(model_inputs,
                                       streamer=streamer,
                                       pad_token_id=2,
                                       max_new_tokens=500,
                                       do_sample=True)
        decoded = tokenizer.batch_decode(generated_ids[:, model_inputs.shape[-1]:], skip_special_tokens=True)
        messages.append({"role": "assistant", "content": decoded[0]})


def get_valid_args(parser):
    args = parser.parse_args()
    return args


def main():
    parser = define_basic_args()
    args = get_valid_args(parser)
    print(args)

    hf_args = {}
    token = os.getenv("HF_TOKEN", None)
    if token is not None:
        hf_args["token"] = token

    model = VQAutoModelQuantization.from_pretrained(args.model, device_map="auto", **hf_args).half()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer or args.model, **hf_args)

    chat_loop(model, tokenizer, args)


main()
