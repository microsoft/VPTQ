import torch
import os
import json
import csv
from pathlib import Path

base_dir = Path("/home/aiscuser/yangwang/vptq_abs_scan/Meta-Llama-3.1-70B-Instruct-abs")

csv_file = base_dir / "model_stats.csv"
headers = ["model_name", "vector_len", "num_centroids", "ppl_wikitext2_2048", "ppl_c4new_2048", 
          "ppl_wikitext2_4096", "ppl_c4new_4096", "ppl_wikitext2_8192", "ppl_c4new_8192"]

with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    
    for folder in sorted(base_dir.glob("2024-*")):
        model_path = folder / "model.pt"
        ppl_path = folder / "ppl_results.json"
        
        if not model_path.exists():
            continue
            
        print(f"Processing {folder.name}...")
        
        try:
            model = torch.load(str(model_path))
            vector_len = model.model.layers[0].self_attn.q_proj.vector_len
            num_centroids = model.model.layers[0].self_attn.q_proj.num_centroids
        except Exception as e:
            print(f"Error loading model {folder.name}: {e}")
            continue
            
        row_data = [folder.name, vector_len, num_centroids]
        
        if ppl_path.exists():
            try:
                with open(ppl_path, 'r') as ppl_file:
                    ppl_data = json.load(ppl_file)
                    row_data.extend([
                        ppl_data["ctx_2048"]["wikitext2"],
                        ppl_data["ctx_2048"]["c4-new"],
                        ppl_data["ctx_4096"]["wikitext2"],
                        ppl_data["ctx_4096"]["c4-new"],
                        ppl_data["ctx_8192"]["wikitext2"],
                        ppl_data["ctx_8192"]["c4-new"]
                    ])
            except Exception as e:
                print(f"Error reading PPL results for {folder.name}: {e}")
                row_data.extend(["N/A"] * 6) 
        else:
            row_data.extend(["N/A"] * 6)
            
        writer.writerow(row_data)

print(f"Results saved to {csv_file}")
