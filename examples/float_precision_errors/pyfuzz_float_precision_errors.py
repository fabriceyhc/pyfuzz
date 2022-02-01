#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 01/30/22 4:44 PM
# @Author  : Fabrice Harel-Canada
# @File    : pyfuzz_float_precision_errors.py

import sys
import os

sys.path.insert(1, os.path.abspath("../.."))

from pyfuzz.fuzzers import *
from pyfuzz.byte_mutations import *
from pyfuzz.fuzz_data_interpreter import *

from float_precision_errors import *

import numpy as np
import pandas as pd

ERROR_THRESHOLD = 0

def PyFuzzFloatPrecisionErrors(data):
    fdi = FuzzedDataInterpreter(data)

    a = np.nan_to_num(fdi.claim_float())
    b = np.nan_to_num(fdi.claim_float())
    c = np.nan_to_num(fdi.claim_float())

    quadratic_comparison(a, b, c, ERROR_THRESHOLD)

if __name__ == "__main__":

    runner = FunctionRunner(PyFuzzFloatPrecisionErrors)

    seed = [bytearray([0]*24)]
    fuzzer = MutationFuzzer(seed, mutator=mutate_bytes)
    results = fuzzer.runs(runner, 1000)

    df = pd.DataFrame(results, columns=["output", "status"])
    print(df.groupby("status").size())
    # print("fuzzer.failure_cases:")
    # print(fuzzer.failure_cases)

    exceptions, counts = np.unique(runner.exceptions, return_counts=True)
    print(pd.DataFrame([counts], columns=exceptions))