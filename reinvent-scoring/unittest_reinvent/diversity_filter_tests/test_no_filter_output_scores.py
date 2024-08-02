from reinvent_scoring.scoring.diversity_filters.curriculum_learning import DiversityFilterParameters
from reinvent_scoring.scoring.diversity_filters.curriculum_learning.diversity_filter import DiversityFilter
from reinvent_scoring.scoring.diversity_filters.curriculum_learning.update_diversity_filter_dto import \
    UpdateDiversityFilterDTO
from unittest_reinvent.diversity_filter_tests.test_tanimoto_similarity_base import BaseTanimotoSimilarity
from unittest_reinvent.fixtures.test_data import ASPIRIN, PROPANE, PENTANE, CELECOXIB, BUTANE, HEXANE, METAMIZOLE
from unittest_reinvent.diversity_filter_tests.fixtures import tanimoto_scaffold_filter_arrangement


class TestNoFilterOutputScores(BaseTanimotoSimilarity):

    def setUp(self):
        super().setUp()

        smiles = [ASPIRIN, PROPANE, PENTANE, CELECOXIB, BUTANE, HEXANE, METAMIZOLE]
        scores = [1.0, 1.0, 1.0, 1.0, 0.7, 0.7, 0.3]
        valid_idx = [0, 1, 2, 3, 4, 5, 6]

        sf_parameters = DiversityFilterParameters(name=self.scaffold_enum.NO_FILTER, minscore=0.5,
                                                  minsimilarity=0.4, bucket_size=1)

        self.scaffold_filter = DiversityFilter(sf_parameters)
        final_summary = tanimoto_scaffold_filter_arrangement(smiles, scores, valid_idx)
        self.update_dto = UpdateDiversityFilterDTO(final_summary, [])

    def test_superfluous_addition(self):
        values = self.scaffold_filter.update_score(self.update_dto)
        self.assertTrue(([1.0, 1.0, 1.0, 1.0, 0.7, 0.7, 0.3] == values).all())
