# VPTQ Quantization Algorithm Tutorial

## Disclaimer
**Early Release Disclaimer**

This code is an early-release version extracted from a complete experimental codebase. Some details still need to be fully revised, so please use and test it cautiously.

**Known Issues:**
1. Some parameter functionalities, such as `enable_perm` and `enable_norm`, are missing and have not been tested.
2. `outlier`, `group_num` are not tested for inference (CUDA implementation is not completed).
3. Quantization time is related to the number of GPUs, the number of k-means clusters, and the number of k-means iterations (`--kiter/--ktol`). You can set the parameters to reduce the quantization time or achieve a better quantization accuracy.
4. We have removed layer-wise fine-tuning for quantization because it costs a lot of disk space to store layer-wise inputs/outputs. We found that end-to-end fine-tuning achieves a similar quantization accuracy.
5. The code of end-to-end fine-tuning is based on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), and we have not released it yet. You can hack it from the source code.

**Contributions Welcome:**
We encourage everyone to submit issues and pull requests to help us improve this project. Your feedback and contributions are greatly appreciated!

Current VPTQ only provides a `coarse-grained` quantization parameter, and you can design a better and fine-grained quantization parameter for research.

## Environment Setting
```bash
# create conda environment
git clone https://github.com/microsoft/VPTQ.git
cd VPTQ
git checkout algorithm


conda env create -f algo-environment.yml
conda activate vptq-algo

# set cuda path for flash_attn
export PATH=/usr/local/cuda-12/bin/:$PATH
pip install flash-attn==2.5.8

# or install VPTQ without compiling to save time
# skip CUDA compilation for algorithm development
SKIP_COMPILE=1 pip install -e . --no-build-isolation

# install VPTQ with cuda 12.1 support for A100 (8.0)
# TORCH_CUDA_ARCH_LIST=8.0 pip install -e . --no-build-isolation

```
## Quantization Example
### Quantization on Meta-Llama-3.1-8B-Instruct
#### 0. (Optional) Download pre-collected hessian/invhessian files
```bash
sudo apt install -y git-lfs
git lfs install
git clone https://huggingface.co/VPTQ-community/Hessians-Llama-31-8B-Instruct-6144-8k
git clone https://huggingface.co/VPTQ-community/InvHessians-Llama-31-8B-Instruct-6144-8k
```

#### 1. Quantization Example: Meta-Llama-3.1-8B-Instruct on 8 GPUs (~3bits)
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run_vptq.py \
        --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
        --output_dir outputs/Meta-Llama-3.1-8B-Instruct/ \
        --vector_lens -1 8 \
        --group_num 1 \
        --num_centroids -1 65536 \
        --num_res_centroids -1 256 \
        --npercent 0 
        --blocksize 128 \
        --new_eval \
        --seq_len 8192 \
        --kmeans_mode hessian \
        --num_gpus 8 \
        --enable_perm \
        --enable_norm \
        --save_model \
        --save_packed_model \
        --hessian_path Hessians-Llama-31-8B-Instruct-6144-8k \
        --inv_hessian_path InvHessians-Llama-31-8B-Instruct-6144-8k \
        --ktol 1e-5 --kiter 100
```

#### 2. **Check the log file for the quantization details**
```bash
cat outputs/Meta-Llama-3.1-8B-Instruct/{your_path}/log/0.log
```


#### 3. **Check packed model**
```bash
python -m vptq --model ./outputs/Meta-Llama-3.1-8B-Instruct/{your_path}/packed_model/ --chat
```

### Quantization on Qwen-2.5-7B-Instruct (~3bits)
#### 0. (Optional) Download pre-collected hessian/invhessian files
```bash
git clone https://huggingface.co/VPTQ-community/Hessians-Qwen2.5-7B-Instruct-6144-8k
git clone https://huggingface.co/VPTQ-community/InvHessians-Qwen2.5-7B-Instruct-6144-8k
```

#### 1. Quantization Example: Qwen-2.5-7B-Instruct on 8 GPUs
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run_vptq.py \
        --model_name Qwen/Qwen2.5-7B-Instruct \
        --output_dir outputs/Qwen2.5-7B-Instruct/ \
        --vector_lens -1 8 \
        --group_num 1 \
        --num_centroids -1 65536 \
        --num_res_centroids -1 256 \
        --npercent 0 \
        --blocksize 128 \
        --new_eval \
        --seq_len 8192 \
        --kmeans_mode hessian \
        --num_gpus 8 \
        --enable_perm \
        --enable_norm \
        --save_model \
        --save_packed_model \
        --hessian_path Hessians-Qwen2.5-7B-Instruct-6144-8k \
        --inv_hessian_path InvHessians-Qwen2.5-7B-Instruct-6144-8k \
        --ktol 1e-5 --kiter 100
```
#### 2. **run inference**
```bash
python run_vptq.py --model ./outputs/Qwen2.5-7B-Instruct/{your_path}/packed_model/ --chat
```


## How to understand the log file?
The log file records the quantization details of each layer, and you can check the execution time and the quantization details of each layer. The most important part is the proxy error before and after quantization, which indicates the quantization accuracy. 
(from QuIP/QuIP#)

Current VPTQ only provides a `coarse-grained` quantization parameter, and you can design a better and fine-grained quantization parameter for research.
```log
INFO - load Hessian from Hessians-Llama-31-8B-Instruct-6144-8k/0_up.pt
INFO - load inv Hessian from InvHessians-Llama-31-8B-Instruct-6144-8k/0_up.pt
INFO - ----Quantizing llama ...---- 2024-10-27 17:05:25 0.mlp.up_proj
INFO - enabling norm dim 0, layer_name:0.mlp.up_proj, scale:torch.Size([4096]), bias:torch.Size([4096])
INFO - kmeans_mode: hessian, enable_perm: hessian
INFO - data shape: torch.Size([14336, 4096]), weights shape: torch.Size([14336, 4096])
INFO - group_size: 4096 number of groups: 1
INFO - idx: 0, num_centroids: -1, skip
INFO - cuml kmeans 101 iterations, error 4701633.5
INFO - idx: 1, quant_data shape: torch.Size([7340032, 8])
INFO - idx: 1, quant_data shape: torch.Size([14336, 4096])
INFO - quantized_data shape: torch.Size([14336, 4096])
INFO - 0.mlp.up_proj 1st kmeans time: 130.61756539344788
INFO - 0.mlp.up_proj qweight init shape: torch.Size([14336, 4096]), weight shape: torch.Size([14336, 4096])
INFO - 0.mlp.up_proj proxy error before VPTQ: 0.2930986285209656, 5.1929931640625, 0.05644117295742035
INFO - 0.mlp.up_proj 1st error time: 5.599343776702881
INFO - 0.mlp.up_proj proxy error after VPTQ: 0.1943642497062683, 5.1929931640625, 0.037428174167871475
INFO - group_size: 4096 number of groups: 1
INFO - idx: 0, num_centroids: -1, skip
INFO - kmeans_mode: hessian, cuml kmeans, 256 clusters
INFO - cuml kmeans 101 iterations, error 1693024.0
INFO - idx: 1, res quant_data shape: torch.Size([14336, 4096])
INFO - 0.mlp.up_proj residual time: 0.9147641658782959
INFO - 0.mlp.up_proj 2ed gptq time: 5.842223405838013
INFO - 0.mlp.up_proj 2ed error time: 0.001397848129272461
INFO - 0.mlp.up_proj proxy error after residual VPTQ: 0.061381880193948746, 5.1929931640625, 0.011820134706795216
```



