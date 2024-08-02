import unittest
from typing import List

from reinvent_scoring.scoring.enums.component_specific_parameters_enum import ComponentSpecificParametersEnum

from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring import CustomProduct
from unittest_reinvent.scoring_tests.fixtures.predictive_model_fixtures import create_activity_component_regression
from reinvent_scoring.scoring.enums import ScoringFunctionComponentNameEnum


class BaseTestPrimaryMultiplicative(unittest.TestCase):

    def init(self, smiles_1: List[str] = [], smiles_2: List[str] = [], smiles_3: List[str] = []):
        self.smiles_1 = smiles_1
        self.smiles_2 = smiles_2
        self.smiles_3 = smiles_3

    def setUp(self):
        enum = ScoringFunctionComponentNameEnum()
        csp_enum = ComponentSpecificParametersEnum()
        activity = create_activity_component_regression()
        qed_score = ComponentParameters(component_type=enum.QED_SCORE,
                                        name="qed_score_name",
                                        weight=1.,
                                        specific_parameters={csp_enum.SMILES:self.smiles_1})
        custom_alerts = ComponentParameters(component_type=enum.CUSTOM_ALERTS,
                                            name="custom_alerts_name",
                                            weight=1.,
                                            specific_parameters={csp_enum.SMILES:self.smiles_2})
        matching_substructure = ComponentParameters(component_type=enum.MATCHING_SUBSTRUCTURE,
                                                    name="matching_substructure_name",
                                                    weight=1.,
                                                    specific_parameters={csp_enum.SMILES:self.smiles_3})
        self.sf_state = CustomProduct(
            parameters=[activity, qed_score, custom_alerts, matching_substructure]
        )
