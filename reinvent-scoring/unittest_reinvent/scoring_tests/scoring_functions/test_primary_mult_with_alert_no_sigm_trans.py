from unittest_reinvent.fixtures.test_data import PENTANE, BENZENE, CELECOXIB
from unittest_reinvent.scoring_tests.scoring_functions.base_test_primary_multiplicative import BaseTestPrimaryMultiplicative


class TestPrimaryMultWithAlertNoSigmTrans(BaseTestPrimaryMultiplicative):

    def setUp(self):
        smiles_2 = [PENTANE]
        smiles_3 = [BENZENE]
        super().init(smiles_2=smiles_2, smiles_3=smiles_3)
        super().setUp()

    def test_primary_mult_with_alert_no_transform_1(self):
        score = self.sf_state.get_final_score(smiles=[PENTANE])
        self.assertEqual(score.total_score, 0)

    def test_primary_mult_with_alert_no_transform_2(self):
        score = self.sf_state.get_final_score(smiles=[CELECOXIB])
        self.assertAlmostEqual(score.total_score[0],  0.337, 3)
