#!/bin/bash

# python3 vptq/tests/test_generation.py 2>&1 | tee test.log

python3 vptq/tests/ops/test_quant_gemv.py 2>&1 | tee test.log
