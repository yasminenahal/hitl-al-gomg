from reinvent_scoring import CustomProduct
from unittest_reinvent.fixtures.test_data import BENZENE, CELECOXIB
from unittest_reinvent.scoring_tests.fixtures import create_predictive_property_component_regression
from unittest_reinvent.scoring_tests.scoring_functions.base_test_selectivity_multiplicative import \
    BaseTestSelectivityMultiplicative


class TestSelectivityMultiplicativeFunctionWithPredictiveProperty(BaseTestSelectivityMultiplicative):

    def setUp(self):
        super().init(predictive_property=create_predictive_property_component_regression(),
                     matching_substructure_smiles=[BENZENE])
        super().setUp()

        self.sf_state = CustomProduct(
            parameters=[self.activity, self.qed_score, self.matching_substructure,
                        self.predictive_property])

    def test_special_selectivity_multiplicative_1(self):
        score = self.sf_state.get_final_score(smiles=[CELECOXIB])
        self.assertAlmostEqual(score.total_score[0],  0.258, 3)
