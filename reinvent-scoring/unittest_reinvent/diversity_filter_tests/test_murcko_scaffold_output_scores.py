from reinvent_scoring.scoring.diversity_filters.curriculum_learning import DiversityFilterParameters
from reinvent_scoring.scoring.diversity_filters.curriculum_learning.diversity_filter import DiversityFilter
from reinvent_scoring.scoring.diversity_filters.curriculum_learning.update_diversity_filter_dto import \
    UpdateDiversityFilterDTO
from unittest_reinvent.diversity_filter_tests.test_murcko_scaffold_base import BaseMurckoScaffoldFilter

from unittest_reinvent.diversity_filter_tests.fixtures import tanimoto_scaffold_filter_arrangement
from unittest_reinvent.fixtures.test_data import PROPANE, BUTANE, PENTANE, HEXANE, ASPIRIN, CELECOXIB, METAMIZOLE


class TestMurckoScaffoldOutputScore(BaseMurckoScaffoldFilter):

    def setUp(self):
        super().setUp()

        # try to add a smile already present
        final_summary = tanimoto_scaffold_filter_arrangement(
        [ASPIRIN, PROPANE, PENTANE, CELECOXIB, BUTANE, HEXANE, METAMIZOLE],
        [1.0, 1.0, 1.0, 1.0, 0.7, 0.7, 0.3], [0, 1, 2, 3, 4, 5, 6])
        self.update_dto = UpdateDiversityFilterDTO(final_summary, [])

        sf_parameters = DiversityFilterParameters(name=self.scaffold_enum.IDENTICAL_MURCKO_SCAFFOLD, minscore=0.5,
                                                  minsimilarity=0.4, bucket_size=1)
        self.scaffold_filter = DiversityFilter(sf_parameters)

    def test_output_scores(self):
        values = self.scaffold_filter.update_score(self.update_dto)

        # assert, that the return values are as expected:
        # - #3, #5 and #6 is set to 0, as the (empty) scaffold is already present and "nbmax" is 1
        # - #7 is stays as 0.3, although below the "minscore" (will not be added to the memory, though)
        self.assertTrue(([1., 1., 0.0, 1., 0.0, 0.0, 0.3] == values).all())
