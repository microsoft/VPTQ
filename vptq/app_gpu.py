import subprocess
import re
import gradio as gr
import plotly.graph_objs as go
from collections import deque

# 用于存储历史数据的队列（保存最近20次的 GPU 利用率和显存使用率）
gpu_util_history = deque(maxlen=20)
mem_usage_history = deque(maxlen=20)


def get_nvidia_smi_info():
    result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, text=True)
    return result.stdout


def parse_nvidia_smi_output(output):
    gpu_info = {}
    utilization = re.search(r'(\d+)%\s+Default', output)
    mem_used = re.search(r'(\d+)MiB / (\d+)MiB', output)
    temp = re.search(r'(\d+)C', output)
    power = re.search(r'(\d+)\s*/\s*(\d+)\s*W', output)
    gpu_clock = re.search(r'(\d+)MHz\s+MEM\s+(\d+)MHz', output)

    if utilization:
        gpu_info['gpu_util'] = int(utilization.group(1))
    if mem_used:
        gpu_info['mem_used'] = int(mem_used.group(1))
        gpu_info['mem_total'] = int(mem_used.group(2))
        gpu_info['mem_percent'] = gpu_info['mem_used'] / gpu_info['mem_total'] * 100
    if temp:
        gpu_info['temp'] = int(temp.group(1))
    if power:
        gpu_info['power_used'] = int(power.group(1))
        gpu_info['power_max'] = int(power.group(2))
    if gpu_clock:
        gpu_info['gpu_clock'] = int(gpu_clock.group(1))
        gpu_info['mem_clock'] = int(gpu_clock.group(2))

    return gpu_info


def update_charts(chart_height=400):  # 添加默认高度参数
    # 获取并解析 nvidia-smi 数据
    output = get_nvidia_smi_info()
    gpu_info = parse_nvidia_smi_output(output)

    # 更新历史数据
    gpu_util = round(gpu_info.get('gpu_util', 0), 1)
    mem_percent = round(gpu_info.get('mem_percent', 0), 1)
    gpu_util_history.append(gpu_util)
    mem_usage_history.append(mem_percent)

    # 创建 GPU 使用率折线图
    gpu_trace = go.Scatter(
        y=list(gpu_util_history),
        mode='lines+markers+text',
        name='GPU Utilization (%)',
        text=list(gpu_util_history),
        textposition='top center'
    )

    # 创建显存使用率折线图
    mem_trace = go.Scatter(
        y=list(mem_usage_history),
        mode='lines+markers+text',
        name='Memory Usage (%)',
        text=list(mem_usage_history),
        textposition='top center'
    )

    # 布局设置，包括标题和注释
    layout = go.Layout(
        # title="Real-time GPU Stats",
        xaxis=dict(
            title=None,
            showticklabels=False,
            ticks=''
        ),
        yaxis=dict(
            title='Percentage (%)',
            range=[-5, 110]  # 调整 y 轴范围
        ),
        height=chart_height,  # 使用传入的高度参数
        margin=dict(l=10, r=10, t=0, b=0)  # 减小上下左右的边距
    )

    # 创建图表
    fig = go.Figure(data=[gpu_trace, mem_trace], layout=layout)
    return fig


def mem_bar(used, total):
    bar_length = 50
    used_bars = int(bar_length * used / total)
    bar = '|' * used_bars + ' ' * (bar_length - used_bars)
    return f"<span style='color: green;'>MEM[{bar}{used:.3f}Gi/{total:.3f}Gi]</span>"


def refresh_gpu_data():
    output = get_nvidia_smi_info()
    gpu_info = parse_nvidia_smi_output(output)

    gpu_clock = gpu_info.get('gpu_clock', 'N/A')
    mem_clock = gpu_info.get('mem_clock', 'N/A')
    temp = gpu_info.get('temp', 'N/A')
    power_used = gpu_info.get('power_used', 'N/A')
    power_max = gpu_info.get('power_max', 'N/A')
    gpu_util = gpu_info.get('gpu_util', 0)
    mem_used = gpu_info.get('mem_used', 0) / 1024  # MiB to GiB
    mem_total = gpu_info.get('mem_total', 0) / 1024  # MiB to GiB

    gpu_info_display = (f"<div style='font-family: monospace;'>"
                        f"<b style='color: yellow;'>Device 0</b> [<span style='color: cyan;'>NVIDIA A100 80GB PCIe</span>] PCIe GEN 4@16x RX: <b>0.000 KiB/s</b> TX: <b>0.000 KiB/s</b><br>"
                        f"GPU <b>{gpu_clock}MHz</b> MEM <b>{mem_clock}MHz</b> TEMP <b style='color: orange;'>{temp}°C</b> FAN <b>N/A%</b> POW <b style='color: red;'>{power_used} / {power_max} W</b><br>"
                        f"GPU[<b>{gpu_util}%</b>] {mem_bar(mem_used, mem_total)}"
                        f"</div>")

    return gpu_info_display


def initialize_history():
    for _ in range(20):
        output = get_nvidia_smi_info()
        gpu_info = parse_nvidia_smi_output(output)
        gpu_util_history.append(round(gpu_info.get('gpu_util', 0), 1))
        mem_usage_history.append(round(gpu_info.get('mem_percent', 0), 1))


# 启动 Gradio 应用
if __name__ == "__main__":
    time_interval = 0.01  # 每 0.1 秒刷新一次
    # Gradio 界面配置
    with gr.Blocks() as demo:
        with gr.Column():
            # 存在闪缩问题，暂时注掉
            # gpu_info_display = gr.HTML(refresh_gpu_data, every=time_interval, elem_id="gpu_info")
            initialize_history()
            gpu_chart = gr.Plot(update_charts, every=time_interval)

    demo.launch()
