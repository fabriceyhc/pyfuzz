#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 01/24/11:09 PM
# @Author  : Fabrice Harel-Canada
# @File    : pyfuzz_minist.py

import sys
import os

sys.path.insert(1, os.path.abspath("../.."))

import mnist as mnist

from pyfuzz.fuzzers import *
from pyfuzz.byte_mutations import *
from pyfuzz.fuzz_data_interpreter import *
 
import torch
import numpy as np
import pandas as pd


def PyFuzzUstableMnist(data):
    fdi = FuzzedDataInterpreter(data)
    img = np.empty([1,784], dtype=float)

    for i in range(784):
        img[0][i] = fdi.claim_probability() # fdi.claim_int() # fdi.claim_probability()

    img = img.reshape(1, 1, 28, 28)
    img = torch.tensor(img).float()
    # img = img / img.max() # used w/ claim_int()
    img = torch.nan_to_num(img)

    prediction = model.predict(img)
    return prediction


if __name__ == "__main__":

    models = mnist.load_models()

    all_results = []
    for model in models:

        model_name = model.__class__.__name__
        print("Fuzzing %s!" % (model_name))

        runner = FunctionRunner(PyFuzzUstableMnist)

        seed = [bytearray([0] * 784 * 4)]
        fuzzer = MutationFuzzer(seed, mutator=mutate_bytes)
        results = fuzzer.runs(runner, 1000000)

        all_results.append(results)

    for model, results in zip(models, all_results):
        model_name = model.__class__.__name__
        print("Fuzzing results for %s!" % (model_name))
        df = pd.DataFrame(results, columns=["output", "status"])
        print(df.groupby("status").size())
        # print("fuzzer.failure_cases:")
        # print(fuzzer.failure_cases)
