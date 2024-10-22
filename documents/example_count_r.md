# VPTQ can count 'r' NOW!

We've just released the latest model, Llama-3.1-Nemotron-70B-Instruct-HF [VPTQ-community](https://huggingface.co/VPTQ-community), compressed from 4 bits to 1.5 bits. Everyone is welcome to try out the new model! We invite all kinds of suggestions. Now, we can count "r" on our local GPUs!

```bash
CUDA_VISIBLE_DEVICES=0 python -m vptq --model VPTQ-community/Llama-3.1-Nemotron-70B-Instruct-HF-v8-k65536-65536-woft --chat
# how many r in strrrraberry
```

![count_r](https://github.com/user-attachments/assets/054b7cf7-2159-4357-9df1-67f3539a407f)
