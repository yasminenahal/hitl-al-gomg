from unittest_reinvent.fixtures.test_data import BUTANE, CELECOXIB, ASPIRIN
from unittest_reinvent.scoring_tests.scoring_functions.base_test_custom_sum import BaseTestCustomSum


class TestTanimotoSimilarity(BaseTestCustomSum):

    def setUp(self):
        self.setup_attrs()
        smiles = [CELECOXIB, BUTANE]
        super().init(self.sf_enum.TANIMOTO_SIMILARITY, "tanimoto_similarity", smiles)
        super().setUp()

    def test_similarity_1(self):
        score = self.sf_state.get_final_score(smiles=[BUTANE])
        self.assertEqual(score.total_score, [1.])

    def test_similarity_2(self):
        score = self.sf_state.get_final_score(smiles=[CELECOXIB])
        self.assertEqual(score.total_score, [1.])

    def test_similarity_3(self):
        score = self.sf_state.get_final_score(smiles=[ASPIRIN])
        self.assertGreater([.5], score.total_score)

    def test_similarity_4(self):
        score = self.sf_state.get_final_score(smiles=[BUTANE, CELECOXIB])
        for i, s in enumerate(score.total_score):
            self.assertEqual(score.total_score[i], 1.)