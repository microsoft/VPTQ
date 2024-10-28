# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import threading

import gradio as gr
from huggingface_hub import snapshot_download

from vptq.app_gpu import disable_gpu_info, enable_gpu_info, update_charts as _update_charts
from vptq.app_utils import get_chat_loop_generator

models = [
    {
        "name": "VPTQ-community/Meta-Llama-3.1-70B-Instruct-v8-k65536-65536-woft",
        "bits": "4 bits"
    },
    {
        "name": "VPTQ-community/Meta-Llama-3.1-70B-Instruct-v8-k65536-256-woft",
        "bits": "3 bits"
    },
    {
        "name": "VPTQ-community/Meta-Llama-3.1-70B-Instruct-v16-k65536-65536-woft",
        "bits": "2 bits"
    },
    {
        "name": "VPTQ-community/Meta-Llama-3.1-70B-Instruct-v8-k32768-0-woft",
        "bits": "1.875 bits"
    },
    {
        "name": "VPTQ-community/Meta-Llama-3.1-8B-Instruct-v8-k65536-65536-woft",
        "bits": "4 bits"
    },
    {
        "name": "VPTQ-community/Meta-Llama-3.1-8B-Instruct-v8-k65536-256-woft",
        "bits": "3 bits"
    },
    {
        "name": "VPTQ-community/Meta-Llama-3.1-8B-Instruct-v12-k65536-4096-woft",
        "bits": "2.3 bits"
    },
    {
        "name": "VPTQ-community/Qwen2.5-72B-Instruct-v8-k65536-65536-woft",
        "bits": "4 bits"
    },
    {
        "name": "VPTQ-community/Qwen2.5-72B-Instruct-v8-k65536-256-woft",
        "bits": "3 bits"
    },
    {
        "name": "VPTQ-community/Qwen2.5-72B-Instruct-v16-k65536-65536-woft",
        "bits": "2 bits"
    },
    {
        "name": "VPTQ-community/Qwen2.5-72B-Instruct-v16-k65536-32768-woft",
        "bits": "1.94 bits"
    },
]

model_choices = [f"{model['name']} ({model['bits']})" for model in models]
display_to_model = {f"{model['name']} ({model['bits']})": model['name'] for model in models}


def download_model(model):
    print(f"Downloading {model['name']}...")
    snapshot_download(repo_id=model['name'])


def download_models_in_background():
    print('Downloading models for the first time...')
    for model in models:
        download_model(model)


loaded_model = None
loaded_model_name = None


def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
    selected_model_display_label,
):
    model_name = display_to_model[selected_model_display_label]

    global loaded_model
    global loaded_model_name

    # Check if the model is already loaded
    if model_name is not loaded_model_name:
        # Load and store the model in the cache
        loaded_model = get_chat_loop_generator(model_name)
        loaded_model_name = model_name

    chat_completion = loaded_model

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
enable_gpu_info()
with gr.Blocks(fill_height=True) as demo:
    with gr.Row():

        def update_chart():
            return _update_charts(chart_height=200)

        # update every 0.1 seconds
        gpu_chart = gr.Plot(update_chart, every=0.1)

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
                gr.Dropdown(
                    choices=model_choices,
                    value=model_choices[0],
                    label="Select Model",
                ),
            ],
        )

if __name__ == "__main__":
    share = os.getenv("SHARE_LINK", None) in ["1", "true", "True"]
    download = os.getenv("DOWNLOAD_MODEL", None) in ["1", "true", "True"]
    if download:
        download_thread = threading.Thread(target=download_models_in_background)
        download_thread.start()
    demo.launch(share=share)
    disable_gpu_info()
