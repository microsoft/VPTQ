# 🚀 VPTQ Now Supports Deepseek R1 (671B) Inference on 4×A100 GPUs!

VPTQ now provides preliminary support for inference with Deepseek R1! With our quantized models, you can efficiently run Deepseek R1 on A100 GPUs, which only support BF16/FP16 formats.

![output](https://github.com/user-attachments/assets/b2e229e0-db6a-4cfd-94fe-4bbf8050457e)

Here's a quick start guide for you:
https://github.com/VPTQ/DeepSeek-V3

______________________________________________________________________

## 📦 Installation

First, install VPTQ and the Deepseek inference demo:

```bash
pip install vptq -U

# Clone and setup inference repository
git clone https://github.com/VPTQ/DeepSeek-V3.git
cd DeepSeek-V3
git checkout vptq
pip install -e .
```

______________________________________________________________________

## 📥 Preparing Models

Download models from HuggingFace:

- **2.x-bit Mixed Quantized Model** (cold experts: 2 bits, w1/w2: 2 bits)
  - Better accuracy
  - Fits strictly into 4×A100 80 GB GPUs (occupies ~78 GB per GPU)

```bash
huggingface-cli download VPTQ-community/deepseek-r1_v_8_k_65536_mixed_mp4 --num-works 32
```

- **Uniform 2-bit Quantized Model**
  - Occupies ~66 GB per GPU

```bash
huggingface-cli download VPTQ-community/deepseek-r1_v_8_k_65536_mixed_mp4 --num-works 32
```

Merge models from safetensors to multi-shard format (`model[0-3]-mp4.safetensors`) for inference:

```bash
python merge_safetensor_folder.py --input-dir path_to_download_model --output-dir path_to_merged_model
```

______________________________________________________________________

## 🚦 Running Inference

Run inference using `torchrun`. Choose the appropriate command based on your hardware capabilities:

- **High CPU and Memory Resources:** (Recommended)

```bash
torchrun --nnodes 1 --nproc-per-node 4 \
    /home/aiscuser/yangwang/DeepSeek-V3-inference/generate.py \
    --ckpt-path /home/aiscuser/yangwang/v_8_mix_w13_k65536_w2_wq_wk_wo_dyn_shared_k_65536_256_mp4/ \
    --config /home/aiscuser/yangwang/DeepSeek-V3-inference/configs/config_671B.json \
    --quantize \
    --quant-config /home/aiscuser/yangwang/v_8_mix_w13_k65536_w2_wq_wk_wo_dyn_shared_k_65536_256_mp4/config.json \
    --interactive \
    --max-new-tokens 65536 \
    --temperature 0.15 \
    --num-load-processes 16
```

- **Limited CPU and Memory Resources:**
  (Initialization takes approximately 2 minutes)

```bash
torchrun --nnodes 1 --nproc-per-node 4 \
    /home/aiscuser/yangwang/DeepSeek-V3-inference/generate.py \
    --ckpt-path /home/aiscuser/yangwang/v_8_mix_w13_k65536_w2_wq_wk_wo_dyn_shared_k_65536_256_mp4/ \
    --config /home/aiscuser/yangwang/DeepSeek-V3-inference/configs/config_671B.json \
    --quantize \
    --quant-config /home/aiscuser/yangwang/v_8_mix_w13_k65536_w2_wq_wk_wo_dyn_shared_k_65536_256_mp4/config.json \
    --interactive \
    --max-new-tokens 65536 \
    --temperature 0.15 \
    --num-load-processes 1
```

______________________________________________________________________

## 🔧 Advanced: Resharding and Merging Models (Optional)

### Resharding

To split a single-shard model into multiple shards for different GPU setups:

```bash
python deepseek_reshard.py \
  --input-model path_to_model0-mp1.safetensors \
  --output-path output_model_mp \
  --input-model-config model0-mp1_config.json \
  --world-size 4
```

### Merging Models

Combine models with varying bitwidths for optimized accuracy:

```bash
python deepseek_merge_kv_shared.py \
  --model_0_path model_0_path \
  --model_1_path model_1_path \
  --output_path output_path \
  --num_shards 4
```

______________________________________________________________________

## ⚠️ Known Issues

- The inference demo does not handle line breaks effectively; input your questions on a single line.
- Initial loading may take around 2 minutes due to layer quantization initialization.
- Quantized models require lower temperatures (~0.1-0.2) for coherent output. Higher values may cause unreadable results due to increased noise.
- NCCL may time out during prolonged periods without data input.
