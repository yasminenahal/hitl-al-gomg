import unittest
from typing import List, Optional

from reinvent_scoring.scoring.enums.component_specific_parameters_enum import ComponentSpecificParametersEnum

from reinvent_scoring.scoring.component_parameters import ComponentParameters
from unittest_reinvent.scoring_tests.fixtures.predictive_model_fixtures import create_custom_alerts_configuration, \
    create_activity_component_regression
from reinvent_scoring.scoring.enums import ScoringFunctionComponentNameEnum


class BaseTestSelectivityMultiplicative(unittest.TestCase):

    def init(self, predictive_property: Optional[ComponentParameters] = None,
             qed_smiles: List[str] = [], matching_substructure_smiles: List[str] = []):
        if predictive_property:
            self.predictive_property = predictive_property
        self.activity = create_activity_component_regression()
        self.qed_smiles = qed_smiles
        self.matching_substructure_smiles = matching_substructure_smiles

    def setUp(self):
        csp_enum = ComponentSpecificParametersEnum()
        self.enum = ScoringFunctionComponentNameEnum()
        self.qed_score = ComponentParameters(component_type=self.enum.QED_SCORE,
                                             name="qed_score_name",
                                             weight=1.,
                                             specific_parameters={csp_enum.SMILES:self.qed_smiles})

        self.custom_alerts = create_custom_alerts_configuration()

        self.matching_substructure = ComponentParameters(component_type=self.enum.MATCHING_SUBSTRUCTURE,
                                                         name="matching_substructure_name",
                                                         weight=1.,
                                                         specific_parameters={csp_enum.SMILES:self.matching_substructure_smiles})
