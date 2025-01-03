from sys import argv
import vptq
import torch
from transformers import AutoTokenizer
from transformers import TextStreamer

# v=12, centroids=8192, 
# results: /datadisk/models/2024-11-29-14-53-40/ppl_results.json
model_path = '/datadisk/models/2024-11-29-14-53-40/model.pt'

# v=10, centroids=1024, 
# results: /datadisk/models/2024-11-29-16-24-24/ppl_results.json
# model_path = '/datadisk/models/2024-11-29-16-24-24/model.pt'

# v=8, centroids=256, 
# results: /datadisk/models/2024-11-29-17-52-26/ppl_results.json 
# model_path = '/datadisk/models/2024-11-29-17-52-26/model.pt'

# v=13, centroids=8192
# results: /datadisk/models/2024-12-01-20-03-49/ppl_results.json
# model_path = '/datadisk/models/2024-12-01-20-03-49/model.pt'

model = torch.load(model_path, weights_only=False)
tokenizer = AutoTokenizer.from_pretrained('Meta-Llama/Meta-Llama-3.1-70B-Instruct')

print('------------')
print(model)
print('------------')

model = model.to('cuda:0')
inputs = "Explain: Do Not Go Gentle into That Good Night"

messages = [{"role": "system", "content": "you are a kind bot"}]
messages.append({"role": "user", "content": inputs})
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
encodeds = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda:0")

print(messages)
print("assistant: ", end='')
for output in model.generate(
    input_ids=encodeds,
    max_new_tokens=128,
    streamer=streamer,
    do_sample=True
):
    current_token = tokenizer.decode(output[0], skip_special_tokens=True)
    print(current_token, end='', flush=True)
