# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import torch.nn as nn


def find_layers(module, layers=None, name=""):
    if layers is None:
        layers = [nn.Linear]
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, layers=layers, name=name + "." + name1 if name != "" else name1))
    return res
