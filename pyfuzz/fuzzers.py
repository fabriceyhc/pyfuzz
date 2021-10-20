import random

from pyfuzz.runners import *

class Fuzzer(object):
    def __init__(self):
        pass

    def fuzz(self):
        """Return fuzz input"""
        return ""

    def run(self, runner=Runner()):
        """Run `runner` with fuzz input"""
        return runner.run(self.fuzz())

    def runs(self, runner=PrintRunner(), trials=10):
        """Run `runner` with fuzz input, `trials` times"""
        # Note: the list comprehension below does not invoke self.run() for subclasses
        # return [self.run(runner) for i in range(trials)]
        outcomes = []
        for i in range(trials):
            outcome = self.run(runner)
            if outcome[1] == runner.FAIL:
                self.failure_cases.append(self.inp)
            outcomes.append(outcome)
        return outcomes

class RandomFuzzer(Fuzzer):
    def __init__(self, 
                 min_length=10, 
                 max_length=100,
                 char_start=32, 
                 char_range=32):
        """Produce strings of `min_length` to `max_length` characters
           in the range [`char_start`, `char_start` + `char_range`]"""
        self.min_length = min_length
        self.max_length = max_length
        self.char_start = char_start
        self.char_range = char_range

    def fuzz(self):
        string_length = random.randrange(self.min_length, self.max_length + 1)
        out = ""
        for i in range(0, string_length):
            out += chr(random.randrange(self.char_start,
                                        self.char_start + self.char_range))
        return out


class MutationFuzzer(Fuzzer):
    def __init__(self, seed, min_mutations=2, max_mutations=10, mutator=lambda x: x):
        self.seed = seed
        self.min_mutations = min_mutations
        self.max_mutations = max_mutations
        self.mutator = mutator
        self.reset()

    def reset(self):
        self.failure_cases = []
        self.population = self.seed
        self.seed_index = 0

    def mutate(self, inp):
        return self.mutator(inp)

    def create_candidate(self):
        candidate = random.choice(self.population)
        iters = random.randint(self.min_mutations, self.max_mutations)
        for i in range(iters):
            candidate = self.mutate(candidate)
        return candidate

    def fuzz(self):
        if self.seed_index < len(self.seed):
            # Still seeding
            self.inp = self.seed[self.seed_index]
            self.seed_index += 1
        else:
            # Mutating
            self.inp = self.create_candidate()
        return self.inp

class MutationCoverageFuzzer(MutationFuzzer):
    def reset(self):
        super().reset()
        self.coverages_seen = set()
        # Now empty; we fill this with seed in the first fuzz runs
        self.population = []

    def run(self, runner):
        """Run function(inp) while tracking coverage.
           If we reach new coverage,
           add inp to population and its coverage to population_coverage
        """
        result, outcome = super().run(runner)
        new_coverage = frozenset(runner.coverage())
        if outcome == Runner.PASS and new_coverage not in self.coverages_seen:
            # We have new coverage
            self.population.append(self.inp)
            self.coverages_seen.add(new_coverage)

        return result


if __name__ == '__main__':

    print("== RandomFuzzer " + "=" * 50)
    random_fuzzer = RandomFuzzer(min_length=20, max_length=20)
    for i in range(10):
        print(random_fuzzer.fuzz())

    print("== MutationLineCoverageFuzzer " + "=" * 50)
    from test_programs import cgi_decode
    from string_mutations import mutate_strings
    from byte_mutations import mutate_bytes
    from runners import *

    seed = ["Hello World"]
    cgi_runner = FunctionLineCoverageRunner(cgi_decode)
    m = MutationCoverageFuzzer(seed, mutator=mutate_strings)
    results = m.runs(cgi_runner, 10000)

    print(m.population)
    print(cgi_runner.coverage())

    print("== MutationBranchCoverageFuzzer " + "=" * 50)
    seed = ["Hello World"]
    cgi_runner = FunctionBranchCoverageRunner(cgi_decode)
    m = MutationCoverageFuzzer(seed, mutator=mutate_strings)
    results = m.runs(cgi_runner, 10)

    print(m.population)
    print(cgi_runner.coverage())