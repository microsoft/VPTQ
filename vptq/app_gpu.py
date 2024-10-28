from collections import deque

import gradio as gr
import plotly.graph_objs as go
import pynvml

# Queues for storing historical data (saving the last 100 GPU utilization and memory usage values)
gpu_util_history = deque(maxlen=100)
mem_usage_history = deque(maxlen=100)


def initialize_nvml():
    """
    Initialize NVML (NVIDIA Management Library).
    """
    pynvml.nvmlInit()


def get_gpu_info():
    """
    Get GPU utilization and memory usage information.

    Returns:
        dict: A dictionary containing GPU utilization and memory usage information.
    """
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assuming a single GPU setup
    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)

    gpu_info = {
        'gpu_util': utilization.gpu,
        'mem_used': memory.used / 1024**2,  # Convert bytes to MiB
        'mem_total': memory.total / 1024**2,  # Convert bytes to MiB
        'mem_percent': (memory.used / memory.total) * 100
    }
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
    gpu_info = get_gpu_info()

    # records the latest GPU utilization and memory usage values
    gpu_util = round(gpu_info.get('gpu_util', 0), 1)
    mem_used = round(gpu_info.get('mem_used', 0) / 1024, 2)  # Convert MiB to GiB
    gpu_util_history.append(gpu_util)
    mem_usage_history.append(mem_used)

    # create GPU utilization line chart
    gpu_trace = go.Scatter(
        y=list(gpu_util_history),
        mode='lines+markers',
        text=list(gpu_util_history),
        line=dict(shape='spline', color='blue'),  # Make the line smooth and set color
        yaxis='y1'  # Link to y-axis 1
    )

    # create memory usage line chart
    mem_trace = go.Scatter(
        y=list(mem_usage_history),
        mode='lines+markers',
        text=list(mem_usage_history),
        line=dict(shape='spline', color='red'),  # Make the line smooth and set color
        yaxis='y2'  # Link to y-axis 2
    )

    # set the layout of the chart
    layout = go.Layout(
        xaxis=dict(title=None, showticklabels=False, ticks=''),
        yaxis=dict(
            title='GPU Utilization (%)',
            range=[-5, 110],
            titlefont=dict(color='blue'),
            tickfont=dict(color='blue'),
        ),
        yaxis2=dict(
            title='Memory Usage (GiB)',
            range=[0, max(24,
                          max(mem_usage_history) + 1)],
            titlefont=dict(color='red'),
            tickfont=dict(color='red'),
            overlaying='y',
            side='right'
        ),
        height=chart_height,  # set the height of the chart
        margin=dict(l=10, r=10, t=0, b=0),  # set the margin of the chart
        showlegend=False  # disable the legend
    )

    fig = go.Figure(data=[gpu_trace, mem_trace], layout=layout)
    return fig


def initialize_history():
    """
    Initializes the GPU utilization and memory usage history.
    """
    for _ in range(100):
        gpu_info = get_gpu_info()
        gpu_util_history.append(round(gpu_info.get('gpu_util', 0), 1))
        mem_usage_history.append(round(gpu_info.get('mem_percent', 0), 1))


def enable_gpu_info():
    pynvml.nvmlInit()


def disable_gpu_info():
    pynvml.nvmlShutdown()


if __name__ == "__main__":
    # Initialize NVML
    initialize_nvml()

    # set the update interval of the GPU information
    time_interval = 0.01
    # create the GPU information display and chart
    with gr.Blocks() as demo:
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

    # Shutdown NVML when the script ends
    pynvml.nvmlShutdown()
