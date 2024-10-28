# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import torch.nn as nn


# find specific layers in a model
def find_layers(module, layers=[nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for old_name, child in module.named_children():
        res.update(find_layers(child, layers=layers, name=name + '.' + old_name if name != '' else old_name))
    return res


def replace_layer(module, name, layer):
    for child_name, child_module in module.named_children():
        if child_name == name:
            setattr(module, child_name, layer)
        else:
            replace_layer(child_module, name, layer)
