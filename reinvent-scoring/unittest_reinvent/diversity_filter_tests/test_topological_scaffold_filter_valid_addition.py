from reinvent_scoring.scoring.diversity_filters.curriculum_learning.update_diversity_filter_dto import \
    UpdateDiversityFilterDTO
from unittest_reinvent.diversity_filter_tests.test_topological_scaffold_filter_base import BaseTopologicalScaffoldFilter
from unittest_reinvent.fixtures.test_data import CELECOXIB
from unittest_reinvent.diversity_filter_tests.fixtures import tanimoto_scaffold_filter_arrangement


class TestTopologicalScaffoldValidAddition(BaseTopologicalScaffoldFilter):

    def setUp(self):
        super().setUp()

        final_summary = tanimoto_scaffold_filter_arrangement([CELECOXIB], [1.0], [0])
        self.update_dto = UpdateDiversityFilterDTO(final_summary, [])

    def test_valid_addition(self):
        self.scaffold_filter.update_score(self.update_dto)
        self.assertEqual(3, self.scaffold_filter._diversity_filter_memory.number_of_scaffolds())
