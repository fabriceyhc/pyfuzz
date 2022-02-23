#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 11/9/21 8:10 PM
# @Author  : Jiyuan Wang
# @File    : pyfuzz_minist.py

import sys
import os

sys.path.insert(1, os.path.abspath("."))

import mnist as mnist

from pyfuzz.fuzzers import *
from pyfuzz.byte_mutations import *
from pyfuzz.fuzz_data_interpreter import *
import numpy as np
import pandas as pd


def PyFuzzMnist(data):
    fdi = FuzzedDataInterpreter(data)
    img = np.empty([1,784],dtype=float)

    for i in range(784):
        img[0][i] = fdi.claim_float()

    print(img)

    prediction = mnist.predict(img_array=img, mnist_model=model)
    return prediction


if __name__ == "__main__":
    model = mnist.set_up()

    runner = FunctionRunner(PyFuzzMnist)

    seed = [bytearray([0] * 784 * 4)]
    fuzzer = MutationFuzzer(seed, mutator=mutate_bytes)
    results = fuzzer.runs(runner, 100)

    df = pd.DataFrame(results, columns=["output", "status"])
    print(df.groupby("status").size())
    print("fuzzer.failure_cases:")
    print(fuzzer.failure_cases)
