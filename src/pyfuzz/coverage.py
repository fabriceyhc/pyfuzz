import sys

class Coverage(object):
    # Trace function
    def traceit(self, frame, event, arg):
        if self.original_trace_function is not None:
            self.original_trace_function(frame, event, arg)

        if event == "line":
            function_name = frame.f_code.co_name
            lineno = frame.f_lineno
            self._trace.append((function_name, lineno))

        return self.traceit

    def __init__(self):
        self._trace = []

    # Start of `with` block
    def __enter__(self):
        self.original_trace_function = sys.gettrace()
        sys.settrace(self.traceit)
        return self

    # End of `with` block
    def __exit__(self, exc_type, exc_value, tb):
        sys.settrace(self.original_trace_function)

    def trace(self):
        """The list of executed lines, as (function_name, line_number) pairs"""
        return self._trace

    def coverage(self):
        """The set of executed lines, as (function_name, line_number) pairs"""
        return set(self.trace())


class BranchCoverage(Coverage):
    def coverage(self):
        """The set of executed line pairs"""
        coverage = set()
        past_line = None
        for line in self.trace():
            # print(line, coverage)
            if past_line is not None:
                coverage.add((past_line, line))
            past_line = line

        return coverage


def population_coverage(population, function, coverage_type="line"):

    if coverage_type == "line":
        c = Coverage()
    elif coverage_type == "branch":
        c = BranchCoverage()
    else:
        raise ValueError("coverage_type of {} not among ['line', 'branch']".format(coverage_type))

    cumulative_coverage = []
    all_coverage = set()

    for s in population:
        with Coverage() as cov:
            try:
                function(s)
            except:
                pass
        all_coverage |= cov.coverage()
        cumulative_coverage.append(len(all_coverage))

    return all_coverage, cumulative_coverage