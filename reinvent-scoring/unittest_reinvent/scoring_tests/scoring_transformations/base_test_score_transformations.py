import unittest

from reinvent_scoring import TransformationTypeEnum, TransformationFactory
from reinvent_scoring.scoring.enums import (
    ComponentSpecificParametersEnum, TransformationParametersEnum)


class BaseTestScoreTransformation(unittest.TestCase):

    def setup_attrs(self):
        self.tt_enum = TransformationTypeEnum()
        self.csp_enum = ComponentSpecificParametersEnum()

    def init(self, added_specific_parameters):
        self.added_specific_parameters = added_specific_parameters

    def setUp(self):
        self.v_list = [12.086, 0.0015, 7.9, 123.264, 77.80, 4.0, 111.12]
        self.factory = TransformationFactory()
        base_specific_parameters = {
            self.csp_enum.TRANSFORMATION: {
                TransformationParametersEnum.LOW: 4
            }
        }

        self.specific_parameters = self._merge_parameters(
            base_specific_parameters, self.added_specific_parameters)

        transform_params = self.specific_parameters.get(self.csp_enum.TRANSFORMATION)
        transform_function = self.factory.get_transformation_function(transform_params)
        self.transformed_scores = transform_function(predictions=self.v_list[:],
                                                     parameters=transform_params)

    def update_parameters(self, parameters):
        self._merge_parameters(self.specific_parameters, parameters)

        transform_params = self.specific_parameters.get(self.csp_enum.TRANSFORMATION)
        transform_function = self.factory.get_transformation_function(transform_params)
        self.transformed_scores = transform_function(predictions=self.v_list[:],
                                                     parameters=transform_params)

    def _merge_parameters(self, parameters, new_parameters):
        for key, value in new_parameters.items():
            if isinstance(value, dict):
                sub_dict = parameters.setdefault(key, {})
                self._merge_parameters(sub_dict, value)
            else:
                parameters[key] = value

        return parameters
