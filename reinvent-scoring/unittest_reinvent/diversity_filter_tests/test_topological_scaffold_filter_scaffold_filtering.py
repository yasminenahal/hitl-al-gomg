from reinvent_scoring.scoring.diversity_filters.curriculum_learning.update_diversity_filter_dto import \
    UpdateDiversityFilterDTO
from unittest_reinvent.diversity_filter_tests.test_topological_scaffold_filter_base import BaseTopologicalScaffoldFilter
from unittest_reinvent.fixtures.test_data import CELECOXIB, METAMIZOLE, BUTANE, HEXANE, COCAINE
from unittest_reinvent.diversity_filter_tests.fixtures import tanimoto_scaffold_filter_arrangement


class TestTopologicalScaffoldFiltering(BaseTopologicalScaffoldFilter):

    def setUp(self):
        super().setUp()

        smiles = [METAMIZOLE, BUTANE, HEXANE, COCAINE]
        scores = [1.0, 0.7, 0.7, 0.3]
        valid_idx = [0, 1, 2, 3]
        final_summary = tanimoto_scaffold_filter_arrangement(smiles, scores, valid_idx)
        self.update_dto = UpdateDiversityFilterDTO(final_summary, [])

    def test_scaffold_filtering(self):
        # 4 input smiles, but the second and third are set to 0 because their scaffold is present already and the last
        # one is not added because its score is too low -> total length of 3 afterwards
        self.scaffold_filter.update_score(self.update_dto)
        self.assertEqual(3, self.scaffold_filter._diversity_filter_memory.number_of_scaffolds())

        # as the order of elements is not defined (or at least arbitrary) in dictionaries, sort the scaffold key strings
        # for the comparison
        self.assertTrue(self.scaffold_filter.get_memory_as_dataframe()["Scaffold"].values.sort() ==
                        ['', CELECOXIB, COCAINE].sort())
