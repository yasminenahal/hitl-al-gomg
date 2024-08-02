from unittest_reinvent.fixtures.test_data import CELECOXIB, ASPIRIN, BENZENE
from unittest_reinvent.scoring_tests.scoring_functions.base_test_primary_multiplicative import BaseTestPrimaryMultiplicative


class TestPrimaryMultiplicativeFunction(BaseTestPrimaryMultiplicative):

    def setUp(self):
        smiles_3 = [BENZENE]
        super().init(smiles_3=smiles_3)
        super().setUp()

    def test_primary_multiplicative_1(self):
        score = self.sf_state.get_final_score(smiles=[CELECOXIB])
        self.assertAlmostEqual(score.total_score[0], 0.337, 3)

    def test_primary_multiplicative_2(self):
        score = self.sf_state.get_final_score(smiles=[ASPIRIN])
        self.assertAlmostEqual(score.total_score[0], 0.288, 3)
