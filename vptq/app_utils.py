# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import argparse
import os
from threading import Thread

import transformers

from vptq.layers.model_base import AutoModelForCausalLM as VQAutoModelQuantization


def define_basic_args():
    parser = argparse.ArgumentParser(
        description="""run a quantized model.
A typical usage is:
    python -m vptq --model [model name] --prompt "Explain: Do Not Go Gentle into That Good Night" \
        [--chat-system-prompt "you are a math teacher."]
 """,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model", type=str, required=True, help="float/float16 model to load, such as [mosaicml/mpt-7b]"
    )
    parser.add_argument("--tokenizer", type=str, default="", help="default same as [model]")
    parser.add_argument("--prompt", type=str, default="once upon a time, there ", help="prompt to start generation")
    parser.add_argument("--chat", action="store_true", help="chat with the model")
    parser.add_argument("--chat-system-prompt", type=str, \
                        default="you are a math teacher.", help="system prompt for chat")
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
    # "you are a math teacher."
    print("============================chat with the model============================")
    print("Press 'exit' to quit")
    messages = [{"role": "system", "content": args.chat_system_prompt}]

    streamer = transformers.TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    while True:
        text = input("You: ")
        if text == "exit" or text == "":
            break
        messages.append({"role": "user", "content": text})
        encodeds = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        model_inputs = encodeds.to(model.device)
        print("assistant: ", end='')
        generated_ids = model.generate(
            model_inputs, streamer=streamer, pad_token_id=2, max_new_tokens=500, do_sample=True
        )
        decoded = tokenizer.batch_decode(generated_ids[:, model_inputs.shape[-1]:], skip_special_tokens=True)
        messages.append({"role": "assistant", "content": decoded[0]})


# for gradio generator function
def get_chat_loop_generator(model_id):
    hf_args = {}
    token = os.getenv("HF_TOKEN", None)
    if token is not None:
        hf_args["token"] = token

    model = VQAutoModelQuantization.from_pretrained(model_id, device_map="auto", **hf_args).half()
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, **hf_args)
    if getattr(tokenizer, "chat_template", None) is None:
        raise Exception("warning: this tokenizer didn't provide chat_template.!!!")

    def chat_loop_generator(
        messages, max_tokens: int, stream: bool = True, temperature: float = 1.0, top_p: float = 1.0
    ):
        print("============================chat with the model============================")
        print("Press 'exit' to quit")

        streamer = transformers.TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        encodeds = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        model_inputs = encodeds.to(model.device)
        generation_kwargs = dict(
            model_inputs,
            streamer=streamer,
            max_new_tokens=max_tokens,
            pad_token_id=2,
            do_sample=True,
            temperature=temperature,
            top_p=top_p
        )
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        for new_text in streamer:
            yield new_text
        thread.join()

    return chat_loop_generator


def get_valid_args(parser):
    args = parser.parse_args()
    return args


def main():
    parser = define_basic_args()
    args = get_valid_args(parser)
    print(args)

    #hf_args = {"dtype": torch.bfloat16}
    hf_args = {}
    token = os.getenv("HF_TOKEN", None)
    if token is not None:
        hf_args["token"] = token

    model = VQAutoModelQuantization.from_pretrained(args.model, device_map="auto", **hf_args)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer or args.model, **hf_args)

    chat_loop(model, tokenizer, args)
