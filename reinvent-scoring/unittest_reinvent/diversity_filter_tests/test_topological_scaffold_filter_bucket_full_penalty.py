from reinvent_scoring.scoring.diversity_filters.curriculum_learning import DiversityFilterParameters
from reinvent_scoring.scoring.diversity_filters.curriculum_learning.diversity_filter import DiversityFilter
from reinvent_scoring.scoring.diversity_filters.curriculum_learning.update_diversity_filter_dto import \
    UpdateDiversityFilterDTO
from unittest_reinvent.diversity_filter_tests.test_topological_scaffold_filter_base import BaseTopologicalScaffoldFilter
from unittest_reinvent.fixtures.test_data import CELECOXIB, METAMIZOLE, BUTANE, HEXANE, ASPIRIN, PROPANE, PENTANE
from unittest_reinvent.diversity_filter_tests.fixtures import tanimoto_scaffold_filter_arrangement

class TestTopologicalScaffoldBucketFullPenalty(BaseTopologicalScaffoldFilter):

    def setUp(self):
        super().setUp()

        smiles = [ASPIRIN, PROPANE, PENTANE, CELECOXIB, BUTANE, HEXANE, METAMIZOLE]
        scores = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        valid_idx = [0, 1, 2, 3, 4, 5, 6]
        final_summary = tanimoto_scaffold_filter_arrangement(smiles, scores, valid_idx)
        self.update_dto = UpdateDiversityFilterDTO(final_summary, [])
        sf_parameters = DiversityFilterParameters(name=self.scaffold_enum.IDENTICAL_TOPOLOGICAL_SCAFFOLD, minscore=0.4,
                                                  minsimilarity=0.4, bucket_size=2)

        self.scaffold_filter = DiversityFilter(sf_parameters)

    def test_bucket_full_penalty(self):
        values = self.scaffold_filter.update_score(self.update_dto)

        # note, that a different value for "nbmax" will not result in a different number of scaffolds, but will change
        # the returned values since every smile that is added to a scaffold beyond this limit will score 0
        self.assertEqual(4, self.scaffold_filter._diversity_filter_memory.number_of_scaffolds())
        self.assertTrue(([1., 1., 1.0, 1., 0.0, 0.0, 1.0] == values).all())
