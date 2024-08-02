from reinvent_scoring import CustomProduct, ComponentParameters
from unittest_reinvent.fixtures.test_data import BENZENE, PROPANE, CELECOXIB, ASPIRIN, GENTAMICIN
from unittest_reinvent.scoring_tests.fixtures import create_predictive_property_component_regression
from unittest_reinvent.scoring_tests.scoring_functions.base_test_selectivity_multiplicative import \
    BaseTestSelectivityMultiplicative


class TestSelectivityMultiplicativeFunctionWithPredictivePropertyNoSigmTrans(BaseTestSelectivityMultiplicative):

    def setUp(self):
        super().init(predictive_property=create_predictive_property_component_regression(),
                     matching_substructure_smiles=[BENZENE])
        super().setUp()

        self.custom_alerts = ComponentParameters(component_type=self.enum.CUSTOM_ALERTS,
                                                 name="custom_alerts_name",
                                                 weight=1.,
                                                 specific_parameters={"smiles":[PROPANE]})

        self.sf_state = CustomProduct(
            parameters=[self.activity, self.qed_score, self.custom_alerts, self.matching_substructure,
                        self.predictive_property])

    def test_special_selectivity_multiplicative_no_sigm_trans_1(self):
        score = self.sf_state.get_final_score(smiles=[CELECOXIB])
        self.assertAlmostEqual(score.total_score[0], 0.258, 3)

    def test_special_selectivity_multiplicative_no_sigm_trans_2(self):
        score = self.sf_state.get_final_score(smiles=[ASPIRIN])
        self.assertAlmostEqual(score.total_score[0], 0.232, 3)

    def test_special_selectivity_multiplicative_no_sigm_trans_4(self):
        score = self.sf_state.get_final_score(smiles=[GENTAMICIN])
        self.assertEqual(score.total_score[0], 0)
