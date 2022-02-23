# @Time    : 12/07/21 1:05 PM
# @Author  : Fabrice Harel-Canada
# @File    : pyfuzz_rick_and_morty_stories.py

import sys
import os

sys.path.insert(1, os.path.abspath("."))

from rick_and_morty_stories import RickAndMortyStories

from pyfuzz.fuzzers import *
from pyfuzz.byte_mutations import *
from pyfuzz.fuzz_data_interpreter import *

import torch
import pandas as pd

def PyFuzz_rick_and_morty_stories(data):
  fdi = FuzzedDataInterpreter(data)

  model_input_size = 50
  vocab_size = len(rm_story_generator.pipeline.tokenizer)

  test_input = [int(fdi.claim_probability() * vocab_size) for _ in range(model_input_size)]
  test_input = rm_story_generator.tokens2text(test_input)
  story = rm_story_generator.generate(test_input)
  
  print(test_input, story)

  return story

if __name__ == "__main__":
    
    rm_story_generator = RickAndMortyStories()

    runner = FunctionRunner(PyFuzz_rick_and_morty_stories)

    seed = [bytearray([0] * 4 * 50)]
    fuzzer = MutationFuzzer(seed, mutator=mutate_bytes)
    results = fuzzer.runs(runner, 1000)

    df = pd.DataFrame(results, columns=["output", "status"])
    print(df.groupby("status").size())
    print("fuzzer.failure_cases:")
    print(fuzzer.failure_cases)

    """
    NLG Failure Cases:

    - runtime errors: same as any other ML pipeline
    - other undesirable:
      - ALL CAPS OUTPUTS (regex check)
      - non-grammatical text (CoLA-trained model)
      - prompt-text discontinuity (topic divergence)
      - semantic diversity collapse (TD)
      - syntactic diversity collapse (SD)
      - lexical diversity collapse (LD)
    """
