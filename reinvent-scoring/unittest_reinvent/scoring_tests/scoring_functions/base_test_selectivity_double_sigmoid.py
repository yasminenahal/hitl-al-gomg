import unittest
from typing import List

from reinvent_scoring import ComponentSpecificParametersEnum, TransformationTypeEnum, ScoringFunctionComponentNameEnum, \
    ComponentParameters, TransformationParametersEnum
from unittest_reinvent.scoring_tests.fixtures import create_activity_component_regression, \
    create_offtarget_activity_component_regression


class BaseTestSelectivityFunctionDoubleSigmoid(unittest.TestCase):

    def init(self, selectivity_smiles: List[str]=None, qed_smiles: List[str]=None,
             matching_substructure_smiles: List[str]=None, custom_alert_smiles: List[str]=None):
        self.selectivity_smiles = selectivity_smiles if selectivity_smiles else []
        self.qed_smiles = qed_smiles if qed_smiles else []
        self.matching_substructure_smiles = matching_substructure_smiles if matching_substructure_smiles else []
        self.custom_alert_smiles = custom_alert_smiles if custom_alert_smiles else []

    def setUp(self):
        self.csp_enum = ComponentSpecificParametersEnum()
        self.transf_type = TransformationTypeEnum()
        self.sf_enum = ScoringFunctionComponentNameEnum()
        self.activity = create_activity_component_regression()
        self.activity_transform_params = {
            TransformationParametersEnum.TRANSFORMATION_TYPE:self.transf_type. DOUBLE_SIGMOID,
            TransformationParametersEnum.COEF_DIV: 100.,
            TransformationParametersEnum.COEF_SI: 150.,
            TransformationParametersEnum.COEF_SE: 150., 
        }
        self.activity.specific_parameters[
            self.csp_enum.TRANSFORMATION].update(self.activity_transform_params)
        self.off_activity = create_offtarget_activity_component_regression()

        self.delta_params = {
            "high": 3.0,
            "k": 0.25,
            "low": 0.0,
            "transformation": True,
            "transformation_type": "sigmoid"
        }

        self.selectivity = ComponentParameters(component_type=self.sf_enum.SELECTIVITY,
                                               name="desirability",
                                               weight=1.,
                                               specific_parameters={
                                                   "activity_model_path": self.activity.specific_parameters.get(self.csp_enum.MODEL_PATH),
                                                   "offtarget_model_path": self.off_activity.specific_parameters.get(self.csp_enum.MODEL_PATH),
                                                   "activity_specific_parameters": self.activity.specific_parameters.copy(),
                                                   "offtarget_specific_parameters": self.off_activity.specific_parameters,
                                                   "delta_transformation_parameters": self.delta_params
                                               })
        csp_enum = ComponentSpecificParametersEnum()

        self.qed_score = ComponentParameters(component_type=self.sf_enum.QED_SCORE,
                                             name="qed_score",
                                             weight=1.,
                                             specific_parameters={csp_enum.SMILES: self.qed_smiles})
        self.matching_substructure = ComponentParameters(component_type=self.sf_enum.MATCHING_SUBSTRUCTURE,
                                                         name="matching_substructure",
                                                         weight=1.,
                                                         specific_parameters={csp_enum.SMILES: self.matching_substructure_smiles})
        self.custom_alerts = ComponentParameters(component_type=self.sf_enum.CUSTOM_ALERTS,
                                                 name="custom_alerts",
                                                 weight=1.,
                                                 specific_parameters={csp_enum.SMILES: self.custom_alert_smiles})
