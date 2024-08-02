from unittest_reinvent.diversity_filter_tests.test_topological_scaffold_filter_base import BaseTopologicalScaffoldFilter


class TestTopologicalScaffoldMemorySize(BaseTopologicalScaffoldFilter):

    def test_dataframe_size(self):
        dataframe = self.scaffold_filter.get_memory_as_dataframe()
        self.assertEqual(2, len(dataframe))
