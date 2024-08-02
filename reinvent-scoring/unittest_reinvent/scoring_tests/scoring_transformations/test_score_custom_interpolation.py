import unittest

from reinvent_scoring import (
    TransformationTypeEnum, ComponentSpecificParametersEnum,
    TransformationFactory, TransformationParametersEnum
)


class TestScoreCustomInterpolation(unittest.TestCase):

    def setUp(self):
        self.tt_enum = TransformationTypeEnum()
        self.csp_enum = ComponentSpecificParametersEnum()
        specific_parameters = {
            self.csp_enum.TRANSFORMATION: {
                TransformationParametersEnum.TRUNCATE_RIGHT: True,
                TransformationParametersEnum.TRUNCATE_LEFT: True,
                TransformationParametersEnum.INTERPOLATION_MAP: [{"origin": 0.0, "destination": 0.0},
                                                                 {"origin": 1.0, "destination": 1.0}],
                TransformationParametersEnum.TRANSFORMATION_TYPE: self.tt_enum.CUSTOM_INTERPOLATION
            }
        }

        transform_params = specific_parameters.get(self.csp_enum.TRANSFORMATION)
        self.factory = TransformationFactory()
        transform_function = self.factory.get_transformation_function(transform_params)
        self.transformed_scores = transform_function(predictions=[-1, 0., 0.3, 0.7, 15],
                                                     parameters=transform_params)

    def test_custom_interpolation(self):
        self.assertListEqual([0, 0, 0.3, 0.7, 1], self.transformed_scores.tolist())
