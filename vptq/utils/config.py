# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json


class _Config:

    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)


class Config:

    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            _args = json.load(f)
            quant_args = _args[0]
            model_args = _args[1]

            self.quant_args = _Config(quant_args)
            self.model_args = _Config(model_args)
