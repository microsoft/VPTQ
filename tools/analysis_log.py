import re
import matplotlib.pyplot as plt
import numpy as np

def parse_log_file(file_path):
    results = {}
    
    with open(file_path, 'r') as f:
        content = f.read()
        
    pattern = r'(\d+\.[a-zA-Z_.]+) proxy error (before|after) VPTQ: ([0-9e.-]+), ([0-9e.-]+), ([0-9e.-]+)'
    matches = re.finditer(pattern, content)
    
    for match in matches:
        operator = match.group(1)
        stage = match.group(2)
        errors = [float(match.group(i)) for i in range(3, 6)]
        
        if operator not in results:
            results[operator] = {'before': [], 'after': []}
        
        results[operator][stage] = errors
    
    return results

def plot_errors(results):
    plt.figure(figsize=(15, 10))
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 15))
    error_names = ['mean error', 'qw*H*qw', 'norm error']
    
    operators = sorted(results.keys())
    x = np.arange(len(operators))
    
    for i in range(3):
        before_errors = [results[op]['before'][i] for op in operators]
        after_errors = [results[op]['after'][i] for op in operators]
        
        axes[i].plot(x, before_errors, 'o-', label='Before VPTQ')
        axes[i].plot(x, after_errors, 'o-', label='After VPTQ')
        
        axes[i].set_title(f'{error_names[i]}')
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(operators, rotation=45, ha='right')
        axes[i].legend()
        axes[i].grid(True)
        
        axes[i].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    plt.savefig('vptq_errors.png')
    plt.close()

log_file = '/home/aiscuser/yangwang/VPTQ.dev/outputs/Meta-Llama-3.1-8B-Instruct-minmax/2025-01-05-20-41-58/logs/0.log'  # 替换为实际的日志文件路径
results = parse_log_file(log_file)
plot_errors(results)