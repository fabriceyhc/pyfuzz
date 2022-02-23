# @Time    : 11/11/21 9:43 PM
# @Author  : Fabrice Harel-Canada
# @File    : pyfuzz_h03_20191644.py

import sys
import os

sys.path.insert(1, os.path.abspath("."))

import h03_20191644 as testee

from pyfuzz.fuzzers import *
from pyfuzz.byte_mutations import *
from pyfuzz.fuzz_data_interpreter import *

import torch
import pandas as pd

def PyFuzzh03_20191644(data):
  fdi = FuzzedDataInterpreter(data)

  model_input_size = 4

  test_input = torch.FloatTensor([
    [fdi.claim_float() for _ in range(model_input_size)]
  ])

  prediction = testee.predict(test_input)

  print(test_input, prediction)

  return prediction

if __name__ == "__main__":
    
    testee.setup()

    runner = FunctionRunner(PyFuzzh03_20191644)

    seed = [bytearray([0] * 12)]
    fuzzer = MutationFuzzer(seed, mutator=mutate_bytes)
    results = fuzzer.runs(runner, 1000)

    df = pd.DataFrame(results, columns=["output", "status"])
    print(df.groupby("status").size())
    print("fuzzer.failure_cases:")
    print(fuzzer.failure_cases)
