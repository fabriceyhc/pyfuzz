import pandas as pd
import titanic

# TODO - These imports will break, we need to work on importing pyfuzz properly. 

from pyfuzz.fuzzers import *
from pyfuzz.byte_mutations import *
from pyfuzz.fuzz_data_interpreter import *

def PyFuzzTitanic(data):
  fdi = FuzzedDataInterpreter(data)

  pclass      = fdi.claim_int(4) 
  sex         = fdi.claim_int(4)
  age         = fdi.claim_float()
  fare        = fdi.claim_float()
  cabin       = fdi.claim_float()
  embarked    = fdi.claim_int(4)
  title       = fdi.claim_int(4)
  family_size = fdi.claim_float()

  print(pclass, sex, age, fare, cabin, embarked, title, family_size)

  test_input = titanic.build_test_input(pclass, sex, age, fare, cabin, embarked, title, family_size)
  prediction = titanic.predict(test_input)
  return prediction

if __name__ == "__main__":
    
    titanic.setup()

    runner = FunctionRunner(PyFuzzTitanic)

    seed = [bytearray([0]*24)]
    fuzzer = MutationFuzzer(seed, mutator=mutate_bytes)
    results = fuzzer.runs(runner, 1000)

    df = pd.DataFrame(results, columns=["output", "status"])
    print(df.groupby("status").size())
    print("fuzzer.failure_cases:")
    print(fuzzer.failure_cases)
