# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

__version__ = "0.0.3"
from vptq.layers import AutoModelForCausalLM, VQuantLinear
__all__ = ["AutoModelForCausalLM", "VQuantLinear"]