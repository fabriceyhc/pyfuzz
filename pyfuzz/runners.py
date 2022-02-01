import os
import subprocess
import sys

from pyfuzz.coverage import *

class Runner(object):
    # Test outcomes
    PASS = "PASS"
    FAIL = "FAIL"
    UNRESOLVED = "UNRESOLVED"

    def __init__(self):
        """Initialize"""
        pass

    def run(self, inp):
        """Run the runner with the given input"""
        return (inp, Runner.UNRESOLVED)

class PrintRunner(Runner):
    def run(self, inp):
        """Print the given input"""
        print(inp)
        return (inp, Runner.UNRESOLVED)


class FunctionRunner(Runner):
    def __init__(self, function):
        """Initialize.  `function` is a function to be executed"""
        self.function = function
        self.exceptions = []

    def run_function(self, inp):
        return self.function(inp)

    def run(self, inp):
        try:
            result = self.run_function(inp)
            outcome = self.PASS
        except Exception as err:
            if err.args:
                try:
                    multiple_exceptions = [type(e).__name__ for e in err.args[0][0]]
                    self.exceptions.extend(multiple_exceptions)
                except:
                    pass
            else:
                self.exceptions.append(type(err).__name__)
            
            template = "Oops, the number {2} exception(s) of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(err).__name__, err.args, self.exceptions.count(type(err).__name__))
            print(message)

            tb = err.__traceback__.tb_next.tb_next.tb_next
            print("Error on line {}".format(tb.tb_lineno))
            print("Error with {}".format(tb.tb_frame.f_code.co_name))

            result = None
            outcome = self.FAIL

        return result, outcome

class FunctionLineCoverageRunner(FunctionRunner):
    def run_function(self, inp):
        with Coverage() as cov:
            try:
                result = super().run_function(inp)
            except Exception as err:
                self._coverage = cov.coverage()
                raise err
        self._coverage = cov.coverage()
        return result

    def coverage(self):
        return self._coverage

class FunctionBranchCoverageRunner(FunctionRunner):
    def run_function(self, inp):
        with BranchCoverage() as cov:
            try:
                result = super().run_function(inp)
            except Exception as err:
                self._coverage = cov.coverage()
                raise err
        self._coverage = cov.coverage()
        return result

    def coverage(self):
        return self._coverage

class ProgramRunner(Runner):
    def __init__(self, program):
        """Initialize.  `program` is a program spec as passed to `subprocess.run()`"""
        self.program = program

    def run_process(self, inp=""):
        """Run the program with `inp` as input.  Return result of `subprocess.run()`."""
        return subprocess.run(self.program,
                              input=inp,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              universal_newlines=True)

    def run(self, inp=""):
        """Run the program with `inp` as input.  Return test outcome based on result of `subprocess.run()`."""
        result = self.run_process(inp)

        if result.returncode == 0:
            outcome = self.PASS
        elif result.returncode < 0:
            outcome = self.FAIL
        else:
            outcome = self.UNRESOLVED

        return (result, outcome)

class BinaryProgramRunner(ProgramRunner):
    def run_process(self, inp=""):
        """Run the program with `inp` as input.  Return result of `subprocess.run()`."""
        return subprocess.run(self.program,
                              input=inp.encode(),
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)