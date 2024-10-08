# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os

import gradio as gr

from vptq.app_utils import get_chat_loop_generator
from app_gpu import update_charts as _update_charts
chat_completion = get_chat_loop_generator("VPTQ-community/Meta-Llama-3.1-70B-Instruct-v8-k32768-0-woft")


def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    response = ""

    for message in chat_completion(
            messages,
            max_tokens=max_tokens,
            stream=True,
            temperature=temperature,
            top_p=top_p,
    ):
        token = message

        response += token
        yield response


"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""
with gr.Blocks(fill_height=True) as demo:
    with gr.Row():
        def update_chart():
            return _update_charts(chart_height=200)
        gpu_chart = gr.Plot(update_chart, every=0.01)  # update every 0.01 seconds

    with gr.Column():
        chat_interface = gr.ChatInterface(
            respond,
            additional_inputs=[
                gr.Textbox(value="You are a friendly Chatbot.", label="System message"),
                gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
                gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
                gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.95,
                    step=0.05,
                    label="Top-p (nucleus sampling)",
                ),
            ],
        )


if __name__ == "__main__":
    share = os.getenv("SHARE_LINK", None) in ["1", "true", "True"]
    demo.launch(share=share)
