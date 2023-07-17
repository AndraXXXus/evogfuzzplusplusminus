import sys
import os
import unittest
from evogfuzz.evogfuzz_class import EvoGFuzz

from evogfuzz_formalizations.calculator import (
    grammar_alhazen as grammar,
    initial_inputs,
    prop,
)
from evogfuzz.oracle import OracleResult
from evogfuzz.fitness_functions import fitness_function_failure as fitness_function
from evogfuzz.settings import *


class TestEvoGFuzz(unittest.TestCase):
    def test_python_version(self):
        self.assertTrue(
            sys.version_info >= (3, 10),
            "Python version does not match the minimum requirement!",
        )

    def test_evogfuzz_initialize(self):
        evogfuzz = EvoGFuzz(
            grammar=grammar,
            oracle=prop,
            inputs=initial_inputs,
            fitness_function=fitness_function,
        )
        evogfuzz._setup()
        self.assertTrue(True)

    def test_evogfuzz_found_exceptions(self):
        evogfuzz = EvoGFuzz(
            grammar=grammar,
            oracle=prop,
            inputs=initial_inputs,
            fitness_function=fitness_function,
        )
        found_exceptions = evogfuzz.fuzz()
        self.assertTrue(
            all([True for inp in found_exceptions if inp.oracle == OracleResult.BUG])
        )


if __name__ == "__main__":
    os.environ["Tournament_Selection_Mode"] = str(TOURNAMENT_SELECTION_MODE)
    unittest.main()
