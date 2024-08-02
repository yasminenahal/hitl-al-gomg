import pandas as pd

from unittest_reinvent.diversity_filter_tests.test_diversity_filter_memory_base import BaseDiversityFilterMemory


class TestDiversityFilterMemoryAdd(BaseDiversityFilterMemory):

    def test_empty(self):
        self.assertFalse(self.mem.smiles_exists(self.smi))
        self.assertEqual(self.mem.number_of_smiles(), 0)

    def test_set_memory(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        self.mem.set_memory(df)

        self.assertEqual(len(self.mem.get_memory()), len(df))
