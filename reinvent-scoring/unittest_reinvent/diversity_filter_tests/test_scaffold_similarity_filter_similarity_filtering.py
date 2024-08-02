from reinvent_scoring.scoring.diversity_filters.curriculum_learning import DiversityFilterParameters
from reinvent_scoring.scoring.diversity_filters.curriculum_learning.diversity_filter import DiversityFilter
from reinvent_scoring.scoring.diversity_filters.curriculum_learning.update_diversity_filter_dto import \
    UpdateDiversityFilterDTO
from unittest_reinvent.diversity_filter_tests.test_tanimoto_similarity_base import BaseTanimotoSimilarity
from unittest_reinvent.fixtures.test_data import CELECOXIB, ASPIRIN, CAFFEINE, METAMIZOLE
from unittest_reinvent.diversity_filter_tests.fixtures import tanimoto_scaffold_filter_arrangement



class TestScaffoldSimilarityFiltering(BaseTanimotoSimilarity):

    def setUp(self):
        super().setUp()

        smiles = [ASPIRIN, CAFFEINE, METAMIZOLE, CELECOXIB]
        scores = [1.0, 1.0, 1.0, 1.0]
        valid_idx = [0, 1, 2, 3]

        final_summary = tanimoto_scaffold_filter_arrangement(smiles, scores, valid_idx)
        self.update_dto = UpdateDiversityFilterDTO(final_summary, [])

        # the fingerprint distances between the resulting murcko scaffolds are:
        #    |   s0  |   s1  |   s2  |   s3  |
        # ---|-------|-------|-------|-------|
        # s0 | 1.000 | 0.293 | 0.342 | 0.308 |
        # s1 |       | 1.000 | 0.847 | 0.422 |
        # s2 |       |       | 1.000 | 0.508 |
        # s3 |       |       |       | 1.000 |

    def _set_sf_parameters(self, minsimilarity: float, bucket_size: int, minscore: int = 0.6):
        sf_parameters = DiversityFilterParameters(name=self.scaffold_enum.SCAFFOLD_SIMILARITY, minscore=minscore,
                                                  minsimilarity=minsimilarity, bucket_size=bucket_size)

        self.scaffold_filter = DiversityFilter(sf_parameters)

    def test_similarity_filtering_1(self):
        # ---------
        # use threshold of 0.4, which should reduce the resulting scaffolds to 2 and penalize the last two smiles
        # as parameter "nbmax" is only one
        self._set_sf_parameters(0.4, 1)
        values = self.scaffold_filter.update_score(self.update_dto)
        self.assertTrue(self.scaffold_filter.get_memory_as_dataframe()["Scaffold"].values.sort() ==
                        [CAFFEINE, ASPIRIN].sort())
        self.assertTrue(([1., 1., 1., 0.0] == values).all())

    def test_similarity_filtering_2(self):
        # ---------
        # use threshold of 0.4, which should reduce the resulting scaffolds to 3 but NOT penalize the third smile
        # as parameter "nbmax" is two
        self._set_sf_parameters(0.45, 2)
        values = self.scaffold_filter.update_score(self.update_dto)
        self.assertTrue(self.scaffold_filter.get_memory_as_dataframe()["Scaffold"].values.sort() ==
                        [CAFFEINE, ASPIRIN, CELECOXIB].sort())
        self.assertTrue(([1., 1., 1.0, 1.0] == values).all())

    def test_similarity_filtering_3(self):
        # ---------
        # use threshold of 0.9, which should NOT reduce the resulting scaffolds and thus NOT penalize the any smile
        # although parameter "nbmax" is only one
        self._set_sf_parameters(0.9, 1)
        values = self.scaffold_filter.update_score(self.update_dto)
        self.assertTrue(self.scaffold_filter.get_memory_as_dataframe()["Scaffold"].values.sort() ==
                        [CAFFEINE, METAMIZOLE, ASPIRIN, CELECOXIB].sort())
        self.assertTrue(([1., 1., 1.0, 1.0] == values).all())
