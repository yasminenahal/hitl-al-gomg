from unittest_reinvent.fixtures.test_data import PROPANE, CELECOXIB, ASPIRIN
from unittest_reinvent.scoring_tests.scoring_functions.base_test_custom_sum import BaseTestCustomSum


class TestJaccardDistance(BaseTestCustomSum):

    def setUp(self):
        self.setup_attrs()
        smiles = [CELECOXIB, PROPANE]
        super().init(self.sf_enum.JACCARD_DISTANCE, "jaccard_distance", smiles)
        super().setUp()

    def test_distance_1(self):
        score = self.sf_state.get_final_score(smiles=[PROPANE])
        self.assertEqual(score.total_score[0], .0)

    def test_distance_2(self):
        score = self.sf_state.get_final_score(smiles=[CELECOXIB])
        self.assertEqual(score.total_score[0], .0)

    def test_distance_3(self):
        score = self.sf_state.get_final_score(smiles=[ASPIRIN])
        self.assertAlmostEqual(score.total_score[0], 0.855, 3)

    def test_distance_4(self):
        score = self.sf_state.get_final_score(smiles=[CELECOXIB, PROPANE])
        for i, s in enumerate(score.total_score):
            self.assertEqual(score.total_score[i], .0)
