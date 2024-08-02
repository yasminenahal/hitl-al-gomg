from unittest_reinvent.diversity_filter_tests.test_tanimoto_similarity_base import BaseTanimotoSimilarity


class TestNoFilterMemorySize(BaseTanimotoSimilarity):

    def test_dataframe_size(self):
        dataframe = self.scaffold_filter.get_memory_as_dataframe()
        self.assertEqual(2, len(dataframe))
