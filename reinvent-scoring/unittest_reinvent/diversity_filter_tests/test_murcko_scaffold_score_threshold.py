from reinvent_scoring.scoring.diversity_filters.curriculum_learning import DiversityFilterParameters
from reinvent_scoring.scoring.diversity_filters.curriculum_learning.diversity_filter import DiversityFilter
from reinvent_scoring.scoring.diversity_filters.curriculum_learning.update_diversity_filter_dto import \
    UpdateDiversityFilterDTO
from unittest_reinvent.diversity_filter_tests.test_murcko_scaffold_base import BaseMurckoScaffoldFilter
from unittest_reinvent.diversity_filter_tests.fixtures import tanimoto_scaffold_filter_arrangement
from unittest_reinvent.fixtures.test_data import PROPANE, BUTANE, PENTANE, HEXANE, ASPIRIN, CELECOXIB, METAMIZOLE


class TestMurckoScaffoldScoreThreshold(BaseMurckoScaffoldFilter):

    def setUp(self):
        super().setUp()

        final_summary = tanimoto_scaffold_filter_arrangement(
            [ASPIRIN, PROPANE, PENTANE, CELECOXIB, BUTANE, HEXANE, METAMIZOLE],
            [1.0, 1.0, 1.0, 1.0, 0.7, 0.7, 0.3], [0, 1, 2, 3, 4, 5, 6])
        self.update_dto = UpdateDiversityFilterDTO(final_summary, [])

        sf_parameters = DiversityFilterParameters(name=self.scaffold_enum.IDENTICAL_MURCKO_SCAFFOLD, minscore=0.3,
                                                  minsimilarity=0.5, bucket_size=1)

        self.scaffold_filter = DiversityFilter(sf_parameters)
        self.scaffold_filter.update_score(self.update_dto)

    def test_score_threshold(self):
        # note, that the default threshold of 0.5 would filter "OCCc1c(C)[n+](cs1)Cc2cnc(C)nc2N" out
        # which would lead to a length of 3 for the scaffold memory; also note, that pure aliphatic
        # chains are condensed to an empty scaffold (string)
        self.assertEqual(4, self.scaffold_filter._diversity_filter_memory.number_of_scaffolds())
