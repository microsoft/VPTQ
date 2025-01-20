# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging
import os
import time

import torch

from vptq.layer_quantizer import layer_quantizer


def setup_logging(log_path, task_id, debug=True):
    # create log directory
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Set up logging to file
    log_file = f'{log_path}/{task_id}.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Configure log format
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Set up logging to console for debug
    if debug:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    return logger


# quantize executer
def quantize_executer(task_id, tasks, args, quant_args, input_queues, output_queues, name2hessian=None, dev=None):
    # TODO: we have to set the device in os environment
    # cuml 23.12 only runs on the CUDA:0
    dev = 'cuda:0' if dev is None else dev
    # attention_mask = None
    # position_ids = torch.arange(args.seq_len).unsqueeze(0).to(dev)

    # logger
    log_path = f'{args.output_dir}/logs/'
    setup_logging(log_path, task_id)
    logger = logging.getLogger()

    logger.info(f'----Quantizing on {dev}----')

    if output_queues is None:
        layer_state_dicts = {}
        layer_qlinear_args = {}

    for (layer_idx, layer) in tasks:
        dtype = next(iter(layer.parameters())).dtype

        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        logger.info(f'----Quantizing layer {layer_idx} ...---- {current_time} on {dev} dtype {dtype}')

        layer, qlinear_args = layer_quantizer(
            args,
            quant_args,
            layer,
            layer_idx,
            logger,
            dev,
            dtype,
            name2hessian=name2hessian
        )

        # send quantized layer to output queue
        layer_state_dict = layer.cpu().state_dict()

        # for 70b/405b models,
        # layer_state_dict is too large for CPU memory
        if args.save_qlinear:
            # view uint16 as int16 to bypass KeyError: torch.uint16
            for key, value in layer_state_dict.items():
                if "indices" in key:
                    layer_state_dict[key] = value.view(torch.int16)

            torch.save(layer_state_dict, f'{args.output_dir}/qlinear_layer_state_{layer_idx}.pt')
            torch.save(qlinear_args, f'{args.output_dir}/qlinear_args_{layer_idx}.pt')
        else:
            if output_queues is None:
                layer_state_dicts[layer_idx] = layer_state_dict
                layer_qlinear_args[layer_idx] = qlinear_args
            else:
                output_queues.put((task_id, layer_idx, layer_state_dict, qlinear_args))

    # for single gpu quantization
    if output_queues is None:
        return layer_state_dicts, layer_qlinear_args
