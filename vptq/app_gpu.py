import re
import subprocess
from collections import deque

import gradio as gr
import plotly.graph_objs as go

# Queues for storing historical data (saving the last 20 GPU utilization and memory usage values)
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


def update_charts(chart_height: int = 200) -> go.Figure:
    """
    Update the GPU utilization and memory usage charts.

    Args:
        chart_height (int, optional): used to set the height of the chart. Defaults to 200.

    Returns:
        plotly.graph_objs.Figure: The updated figure containing the GPU and memory usage charts.
    """
    # obtain GPU information
    output = get_nvidia_smi_info()
    gpu_info = parse_nvidia_smi_output(output)

    # records the latest GPU utilization and memory usage values
    gpu_util = round(gpu_info.get('gpu_util', 0), 1)
    mem_percent = round(gpu_info.get('mem_percent', 0), 1)
    gpu_util_history.append(gpu_util)
    mem_usage_history.append(mem_percent)

    # create GPU utilization line chart
    gpu_trace = go.Scatter(y=list(gpu_util_history),
                           mode='lines+markers+text',
                           name='GPU Utilization (%)',
                           text=list(gpu_util_history),
                           textposition='top center')

    # create memory usage line chart
    mem_trace = go.Scatter(y=list(mem_usage_history),
                           mode='lines+markers+text',
                           name='Memory Usage (%)',
                           text=list(mem_usage_history),
                           textposition='top center')

    # set the layout of the chart
    layout = go.Layout(
        # title="Real-time GPU Stats",
        xaxis=dict(title=None, showticklabels=False, ticks=''),
        yaxis=dict(
            title='Percentage (%)',
            range=[-5, 110]  # adjust the range of the y-axis
        ),
        height=chart_height,  # set the height of the chart
        margin=dict(l=10, r=10, t=0, b=0)  # set the margin of the chart
    )

    fig = go.Figure(data=[gpu_trace, mem_trace], layout=layout)
    return fig


def mem_bar(used: float, total: float) -> str:
    """
    Generates a memory usage bar.

    Args:
        used (float): The amount of memory used in GiB.
        total (float): The total amount of memory available in GiB.
    Returns:
        str: A string representing the memory usage bar in HTML format.
    """
    bar_length = 50
    used_bars = int(bar_length * used / total)
    bar = '|' * used_bars + ' ' * (bar_length - used_bars)
    return f"<span style='color: green;'>MEM[{bar}{used:.3f}Gi/{total:.3f}Gi]</span>"


def refresh_gpu_data():
    """
    Refreshes and returns the current GPU data in an HTML formatted string.

    Returns:
        str: An HTML formatted string containing the GPU information, including 
             GPU clock speed, memory clock speed, temperature, power usage, 
             GPU utilization, and memory usage.
    """

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
                        f"<b style='color: yellow;'>Device 0</b> "
                        f"[<span style='color: cyan;'>NVIDIA A100 80GB PCIe</span>] "
                        f"PCIe GEN 4@16x RX: <b>0.000 KiB/s</b> TX: <b>0.000 KiB/s</b><br>"
                        f"GPU <b>{gpu_clock}MHz</b> MEM <b>{mem_clock}MHz</b> "
                        f"TEMP <b style='color: orange;'>{temp}°C</b> FAN <b>N/A%</b> "
                        f"POW <b style='color: red;'>{power_used} / {power_max} W</b><br>"
                        f"GPU[<b>{gpu_util}%</b>] {mem_bar(mem_used, mem_total)}"
                        f"</div>")

    return gpu_info_display


def initialize_history():
    """
    Initializes the GPU utilization and memory usage history.
    """
    for _ in range(20):
        output = get_nvidia_smi_info()
        gpu_info = parse_nvidia_smi_output(output)
        gpu_util_history.append(round(gpu_info.get('gpu_util', 0), 1))
        mem_usage_history.append(round(gpu_info.get('mem_percent', 0), 1))


if __name__ == "__main__":
    # set the update interval of the GPU information
    time_interval = 0.01
    # create the GPU information display and chart
    with gr.Blocks() as demo:
        # Flickering issue exists, temporarily commented out
        gpu_info_display = gr.HTML(refresh_gpu_data, every=time_interval, elem_id="gpu_info")
        initialize_history()
        gpu_chart = gr.Plot(update_charts, every=time_interval)
    # avoid the up and down movement of the GPU information
    demo.css = """
        #gpu_info {
            height: 100px;
            overflow: hidden;
        }
    """
    demo.launch()
