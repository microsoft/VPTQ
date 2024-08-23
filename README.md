# VPTQ: Extreme Low-bit Vector Post-Training Quantization for Large Language Models

> This repo is the official implementation of the paper "VPTQ: Extreme Low-bit Vector Post-Training Quantization for Large Language Models".

Scaling model size significantly challenges the deployment and inference of Large Language Models (LLMs). Due to the redundancy in LLM weights, recent research has focused on pushing weight-only quantization to extremely low-bit (even down to 2 bits). It reduces memory requirements, optimizes storage costs, and decreases memory bandwidth needs during inference. However, due to numerical representation limitations, traditional scalar-based weight quantization struggles to achieve such extreme low-bit. Recent research on Vector Quantization (VQ) for LLMs has demonstrated the potential for extremely low-bit model quantization by compressing vectors into indices using lookup tables.


## Dependencies
- python 3.10+
- torch >= 2.2.0
- transformers >= 4.44.0
- Accelerate >= 0.33.0
- latest datasets

## Installation
```
pip install git+https://github.com/microsoft/VPTQ.git --no-build-isolation
```

## Language Generation
To generate text using the pretrained model, you can use the following code snippet:

`python -m vptq --model=LLaMa-2-7b-1.5bi-vptq --prompt="Hello, my dog is cute"`

Lunching a chatbot:
note that you must use a chat model for this to work
``` 
python -m vptq --model=LLaMa-2-7b-chat-1.5b-vptq --chat
```
Using the python API:
```
import vptq
import transformers
tokenizer = transformers.AutoTokenizer.from_pretrained("LLaMa-2-7b-1.5bi-vptq")
m = vptq.AutoModelForCausalLM.from_pretrained("LLaMa-2-7b-1.5bi-vptq", device_map='auto')

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt").to("cuda")
out = m.generate(**inputs, max_new_tokens=100, pad_token_id=2)
print(tokenizer.decode(out[0], skip_special_tokens=True))

```


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
