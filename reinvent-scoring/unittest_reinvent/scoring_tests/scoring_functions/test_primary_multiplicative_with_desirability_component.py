import unittest

from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring import CustomProduct
from reinvent_scoring.scoring.score_summary import FinalSummary
from unittest_reinvent.scoring_tests.fixtures.predictive_model_fixtures import create_activity_component_regression, \
    create_offtarget_activity_component_regression, create_custom_alerts_configuration
from reinvent_scoring.scoring.enums import ComponentSpecificParametersEnum
from reinvent_scoring.scoring.enums import ScoringFunctionComponentNameEnum
from reinvent_scoring.scoring.enums import TransformationParametersEnum
from reinvent_scoring.scoring.enums import TransformationTypeEnum
from unittest_reinvent.fixtures.test_data import CELECOXIB, METHOXYHYDRAZINE, BENZENE


class TestPrimaryMultiplicativeWithDesirabilityComponent(unittest.TestCase):

    def setUp(self):
        csp_enum = ComponentSpecificParametersEnum()
        transf_type = TransformationTypeEnum()
        enum = ScoringFunctionComponentNameEnum()
        activity = create_activity_component_regression()
        activity_transform_params = {
            TransformationParametersEnum.TRANSFORMATION_TYPE: transf_type.DOUBLE_SIGMOID,
            TransformationParametersEnum.COEF_DIV: 100.,
            TransformationParametersEnum.COEF_SI: 150.,
            TransformationParametersEnum.COEF_SE: 150.,
        }
        activity.specific_parameters[csp_enum.TRANSFORMATION].update(
            activity_transform_params)
        off_activity = create_offtarget_activity_component_regression()

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
                                              "activity_model_path": activity.specific_parameters[csp_enum.MODEL_PATH],
                                              "offtarget_model_path": off_activity.specific_parameters[csp_enum.MODEL_PATH],
                                              "activity_specific_parameters": activity.specific_parameters.copy(),
                                              "offtarget_specific_parameters": off_activity.specific_parameters,
                                              "delta_transformation_parameters": delta_params
                                          })
        qed_score = ComponentParameters(component_type=enum.QED_SCORE,
                                        name="qed_score",
                                        weight=1.,
                                        specific_parameters={})
        matching_substructure = ComponentParameters(component_type=enum.MATCHING_SUBSTRUCTURE,
                                                    name="matching_substructure",
                                                    weight=1.,
                                                    specific_parameters={"smiles":[BENZENE]})
        custom_alerts = create_custom_alerts_configuration()

        self.sf_state = CustomProduct(
            parameters=[activity, selectivity, qed_score, matching_substructure, custom_alerts])

    def test_desirability_component_1(self):
        score: FinalSummary = self.sf_state.get_final_score(smiles=[METHOXYHYDRAZINE])
        self.assertAlmostEqual(score.total_score[0], 0)

    def test_desirability_component_2(self):
        score: FinalSummary = self.sf_state.get_final_score(smiles=[CELECOXIB])
        self.assertAlmostEqual(score.total_score[0], 0.339, 3)
