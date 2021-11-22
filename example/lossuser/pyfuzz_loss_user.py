#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 11/9/21 3:41 PM
# @Author  : Jiyuan Wang
# @File    : pyfuzz_loss_user.py
import example.lossuser.loss_user as lo
import pandas as pd

from pyfuzz.fuzzers import *
from pyfuzz.byte_mutations import *
from pyfuzz.fuzz_data_interpreter import *

def PyLossuser(data):
  fdi = FuzzedDataInterpreter(data)

  Partner           = fdi.claim_int(4)
  Dependents        = fdi.claim_int(4)
  SeniorCitizen     = fdi.claim_int(4)
  MultipleLines_Yes = fdi.claim_int(4)
  TechSupport_Yes   = fdi.claim_int(4)
  Contract_Oneyear  = fdi.claim_int(4)
  Contract_Twoyear  = fdi.claim_int(4)
  tenure            = fdi.claim_int(4)
  MonthlyCharges    = fdi.claim_float()
  TotalCharges      = fdi.claim_float()

  print(SeniorCitizen,Partner,Dependents,MultipleLines_Yes,TechSupport_Yes,
                                   Contract_Oneyear,Contract_Twoyear,tenure,MonthlyCharges,TotalCharges)

  test_input = lo.build_test_input(SeniorCitizen,Partner,Dependents,MultipleLines_Yes,TechSupport_Yes,
                                   Contract_Oneyear,Contract_Twoyear,tenure,MonthlyCharges,TotalCharges)
  prediction = lo.run_model(test_input)
  return prediction


if __name__ == '__main__':
    lo.set_up()
    runner = FunctionRunner(PyLossuser)
    pd.set_option('max_colwidth', 100)
    pd.set_option('display.max_columns', None)
    seed = [bytearray([0] * 36)]
    fuzzer = MutationFuzzer(seed, mutator=mutate_bytes)
    results = fuzzer.runs(runner, 1000)

    df = pd.DataFrame(results, columns=["output", "status"])
    print(df.groupby("status").size())
    print("fuzzer.failure_cases:")
    print(fuzzer.failure_cases)