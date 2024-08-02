import unittest

from reinvent_scoring.scoring.enums.component_specific_parameters_enum import ComponentSpecificParametersEnum

from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring import CustomProduct
from unittest_reinvent.scoring_tests.fixtures.predictive_model_fixtures import create_activity_component_regression, \
    create_predictive_property_component_regression, create_offtarget_activity_component_regression
from reinvent_scoring.scoring.enums import ScoringFunctionComponentNameEnum
from unittest_reinvent.fixtures.test_data import CELECOXIB, ASPIRIN


class TestParallelSelectivityMultiplicativeFunction(unittest.TestCase):

    def setUp(self):
        self.csp_enum = ComponentSpecificParametersEnum()
        enum = ScoringFunctionComponentNameEnum()
        activity = create_activity_component_regression()
        predictive_property1 = create_predictive_property_component_regression()
        predictive_property2 = create_offtarget_activity_component_regression()

        delta_params = {
            "high": 3.0,
            "k": 0.25,
            "low": 0.0,
            "transformation": True,
            "transformation_type": "sigmoid"
        }

        selectivity = ComponentParameters(component_type=enum.SELECTIVITY,
                                          name="desirability",
                                          weight=1.,
                                          specific_parameters={
                                              "activity_model_path": activity.specific_parameters[self.csp_enum.MODEL_PATH],
                                              "offtarget_model_path": predictive_property2.specific_parameters[self.csp_enum.MODEL_PATH],
                                              "activity_specific_parameters": activity.specific_parameters.copy(),
                                              "offtarget_specific_parameters": predictive_property2.specific_parameters,
                                              "delta_transformation_parameters": delta_params
                                          })

        qed_score = ComponentParameters(component_type=enum.QED_SCORE,
                                        name="qed_score_name",
                                        weight=1.,
                                        specific_parameters={})
        custom_alerts = ComponentParameters(component_type=enum.CUSTOM_ALERTS,
                                            name="custom_alerts_name",
                                            weight=1.,
                                            specific_parameters={})
        matching_substructure = ComponentParameters(component_type=enum.MATCHING_SUBSTRUCTURE,
                                                    name="matching_substructure_name",
                                                    weight=1.,
                                                    specific_parameters={self.csp_enum.SMILES: ["c1ccccc1"]})

        self.sf_instance = CustomProduct(parameters=[activity, qed_score, custom_alerts,
                                                     matching_substructure, predictive_property1,
                                                     selectivity], parallel=True)

    def test_selectivity_multiplicative_1(self):
        smiles = [CELECOXIB for _ in range(2)]
        score = self.sf_instance.get_final_score(smiles=smiles)
        self.assertAlmostEqual(score.total_score[0], 0.174, 3)

    def test_selectivity_multiplicative_2(self):
        smiles = [ASPIRIN for _ in range(2)]
        score = self.sf_instance.get_final_score(smiles=smiles)
        self.assertAlmostEqual(score.total_score[0],  0.161, 3)
