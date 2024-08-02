from reinvent_scoring.scoring.diversity_filters.curriculum_learning.update_diversity_filter_dto import \
    UpdateDiversityFilterDTO
from unittest_reinvent.diversity_filter_tests.test_murcko_scaffold_base import BaseMurckoScaffoldFilter
from unittest_reinvent.diversity_filter_tests.fixtures import tanimoto_scaffold_filter_arrangement
from unittest_reinvent.fixtures.test_data import BUTANE, HEXANE, CELECOXIB, CAFFEINE, ASPIRIN, COCAINE


class TestMurckoScaffoldFiltering(BaseMurckoScaffoldFilter):

    def setUp(self):
        super().setUp()

        final_summary = tanimoto_scaffold_filter_arrangement(
            [CELECOXIB, BUTANE, HEXANE, CAFFEINE], [1.0, 0.7, 0.7, 0.3], [0, 1, 2, 3])

        self.update_dto = UpdateDiversityFilterDTO(final_summary, [])

    def test_scaffold_filtering(self):
        # 4 input smiles, but the second and third are set to 0 because their scaffold is present already and the last
        # one is not added because its score is too low -> total length of 3 afterwards
        self.scaffold_filter.update_score(self.update_dto)
        self.assertEqual(3, self.scaffold_filter._diversity_filter_memory.number_of_scaffolds())

        # as the order of elements is not defined (or at least arbitrary) in dictionaries, sort the scaffold key strings
        # for the comparison
        self.assertTrue(self.scaffold_filter.get_memory_as_dataframe()["Scaffold"].values.sort() ==
                        ['', ASPIRIN, COCAINE].sort())