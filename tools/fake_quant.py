import os
import vptq
import torch
import argparse
import torch.multiprocessing as mp
from transformers import AutoTokenizer
from transformers import TextStreamer
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Fake quantization for VPTQ models')
    parser.add_argument('--base_model_path', type=str, required=True,
                      help='Base path to the model directory')
    parser.add_argument('--base_model_name', type=str, required=True,
                      help='Base name of the model')
    parser.add_argument('--tokenizer_path', type=str, required=True,
                      help='Path to the tokenizer')
    parser.add_argument('--output_base_path', type=str, default='.',
                      help='Base path for output models (default: current directory)')
    parser.add_argument('--num_gpus', type=int, default=8,
                      help='Number of GPUs to use (default: 8)')
    return parser.parse_args()

def replace_vqlinear(module, current_key_name):
    global vec_len
    for name, child in module.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)
        layer_name = ".".join(current_key_name)
        
        if isinstance(child, vptq.layers.vqlinear.VQuantLinear):
            print(layer_name)
            setattr(
                module,
                name,
                torch.nn.Linear(
                    child.in_features,
                    child.out_features,
                    child.bias is not None,
                    device="cpu",
                    dtype=child.centroids.weight.dtype,
                ),
            )
            if child.bias is not None:
                getattr(module, name).bias.data = child.bias
            getattr(module, name).weight.data = child.cuda().dequant().cpu()
            vec_len = child.vector_len
            child.cpu()
            torch.cuda.empty_cache()
        else:
            replace_vqlinear(child, current_key_name)
        current_key_name.pop(-1)

def process_model(model_path, tokenizer, base_model_name, output_base_path, gpu_id):
    try:
        print(f"Processing model: {model_path} on GPU {gpu_id}")
        # Set the GPU device for this process
        torch.cuda.set_device(gpu_id)
        
        model = torch.load(str(model_path), weights_only=False)
        replace_vqlinear(model, None)
        
        date_str = model_path.parent.parts[-1]
        output_path = Path(output_base_path) / f"{base_model_name}-{date_str}-{vec_len}_model"
        
        print(f"Vector length: {vec_len}, saving to {output_path}")
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        print(f"Model processing complete on GPU {gpu_id}")
        
    except Exception as e:
        print(f"Error processing {model_path} on GPU {gpu_id}: {str(e)}")
    finally:
        # Clean up GPU memory
        torch.cuda.empty_cache()

def process_batch(args_list):
    model_path, tokenizer_path, base_model_name, output_base_path, gpu_id = args_list
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    process_model(model_path, tokenizer, base_model_name, output_base_path, gpu_id)

def main():
    args = parse_args()
    model_paths = Path(args.base_model_path)
    os.makedirs(args.output_base_path, exist_ok=True)
    
    # Get all valid model paths
    model_path_list = []
    for date_path in model_paths.iterdir():
        model_path = date_path / "model.pt"
        if model_path.exists():
            model_path_list.append(model_path)
        else:
            print(f"Skipping: {model_path} does not exist")
    
    # Prepare arguments for parallel processing
    process_args = []
    for i, model_path in enumerate(model_path_list):
        gpu_id = i % args.num_gpus
        process_args.append((
            model_path,
            args.tokenizer_path,
            args.base_model_name,
            args.output_base_path,
            gpu_id
        ))
    
    # Initialize multiprocessing with spawn method
    mp.set_start_method('spawn', force=True)
    
    # Create pool and process models in parallel
    with mp.Pool(args.num_gpus) as pool:
        pool.map(process_batch, process_args)

if __name__ == "__main__":
    main()