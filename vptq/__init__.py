# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import importlib.metadata

from vptq.layers import AutoModelForCausalLM, VQuantLinear

__version__ = importlib.metadata.version("vptq")

__all__ = ["AutoModelForCausalLM", "VQuantLinear"]
