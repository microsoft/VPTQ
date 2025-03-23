#!/bin/bash

python vptq/tests/ops/test_quant_gemv.py 2>&1 | tee test.log
