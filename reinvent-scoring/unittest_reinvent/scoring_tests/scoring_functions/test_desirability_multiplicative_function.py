from reinvent_scoring.scoring import CustomProduct
from unittest_reinvent.scoring_tests.fixtures.predictive_model_fixtures import create_custom_alerts_configuration

from unittest_reinvent.fixtures.test_data import CELECOXIB, HEXANE, BUTANE, ANILINE, BENZENE
from unittest_reinvent.scoring_tests.scoring_functions.base_test_selectivity_double_sigmoid import \
    BaseTestSelectivityFunctionDoubleSigmoid


class TestDesirabilityMultiplicativeFunction(BaseTestSelectivityFunctionDoubleSigmoid):

    def setUp(self):
        smiles = [BENZENE]
        super().init(matching_substructure_smiles=smiles)
        super().setUp()

        custom_alerts = create_custom_alerts_configuration()

        self.sf_state = CustomProduct(
            parameters=[self.activity, self.selectivity, custom_alerts, self.qed_score, self.matching_substructure]
        )

    def test_desirability_multiplicative_1(self):
        score = self.sf_state.get_final_score(smiles=[BUTANE])
        self.assertAlmostEqual(score.total_score[0], 0.141, 3)

    def test_desirability_multiplicative_2(self):
        score = self.sf_state.get_final_score(smiles=[ANILINE])
        self.assertAlmostEqual(score.total_score[0], 0.292, 3)

    def test_desirability_multiplicative_3(self):
        score = self.sf_state.get_final_score(smiles=[CELECOXIB])
        self.assertAlmostEqual(score.total_score[0], 0.339, 3)

    def test_desirability_multiplicative_4(self):
        score = self.sf_state.get_final_score(smiles=[HEXANE, "12"])
        self.assertAlmostEqual(score.total_score[0], 0.144, 3)
        self.assertEqual(score.total_score[1], 0)
