import unittest
from unittest import mock

from reinvent_scoring import ScoringFunctionComponentNameEnum, TransformationTypeEnum, TransformationParametersEnum
from reinvent_scoring.scoring.enums import ComponentSpecificParametersEnum
from reinvent_scoring.scoring.predictive_model.model_container import ModelContainer
from reinvent_scoring.scoring.score_components import PredictivePropertyComponent, ComponentParameters


class BaseTestPredictivePropertyComponent(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        csp_enum = ComponentSpecificParametersEnum()
        params = {
            csp_enum.TRANSFORMATION: { 
                TransformationParametersEnum.TRANSFORMATION_TYPE: TransformationTypeEnum().NO_TRANSFORMATION,
            },
            csp_enum.SCIKIT: "regression",
            csp_enum.DESCRIPTOR_TYPE: None,
            "container_type": "optuna_container"
        }
        with mock.patch(
            'reinvent_scoring.scoring.score_components.PredictivePropertyComponent._load_container',
            return_value=ModelContainer(cls.model, params)
        ):
            cls.component = PredictivePropertyComponent(
                ComponentParameters(
                    component_type=ScoringFunctionComponentNameEnum.PREDICTIVE_PROPERTY,
                    name="predictive_property",
                    weight=1.,
                    specific_parameters=params
                )
            )
