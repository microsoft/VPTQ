# VPTQ Quantization Algorithm Tutorial

## Environment Setting
```bash
# create conda environment
conda env create -f algo-environment.yml
# install VPTQ without compiling to save time
git clone https://github.com/microsoft/VPTQ.git
cd VPTQ
conda activate vptq-algo
SKIP_COMPILE=1 pip install -e . --no-build-isolation
```

## Quantization Example
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python np_vptq.py \
        --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
        --output_dir Experiments/Meta-Llama-3.1-Instruct-8B/ \
        --vector_lens -1 8 \
        --num_centroids -1 256 \
        --num_res_centroids -1 256 \
        --npercent 1 --blocksize 128 \
        --new_eval --enable_transpose --transpose_all \
        --seq_len 8192 --kmeans_mode hessian \
        --num_gpus 8 --group_num 1 \
        --hessian_path /home/aiscuser/yangwang/Hessians-Llama-31-70B-Instruct-6144-8k \
        --inv_hessian_path /home/aiscuser/yangwang/InvHessians-Llama-31-70B-Instruct-6144-8k \
        --enable_norm --eval_quant --save_model --ktol 1e-5 --kiter 100
```

