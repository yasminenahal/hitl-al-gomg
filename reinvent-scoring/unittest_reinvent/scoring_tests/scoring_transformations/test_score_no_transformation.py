import unittest

from reinvent_scoring.scoring.score_transformations import TransformationFactory
from reinvent_scoring.scoring.enums import ComponentSpecificParametersEnum
from reinvent_scoring.scoring.enums import TransformationTypeEnum, TransformationParametersEnum
from unittest_reinvent.scoring_tests.scoring_transformations.fixtures import round_list


class TestScoreNoTransformation(unittest.TestCase):

    def setUp(self):
        self.tt_enum = TransformationTypeEnum()
        self.csp_enum = ComponentSpecificParametersEnum()
        self.expected_scores = [12.086, 0.002, 7.9, 123.264, 77.80, 4.0, 111.12]
        self.factory = TransformationFactory()

        # ---------
        # note, that the case where "TRANSFORMATION" is set to "False" is not handled here,
        # as this functionality is part of the model container, rather than the transformation
        # factory; however, the expected result is the same as for "test_no_transformation"
        # ---------
        specific_parameters = {
            self.csp_enum.TRANSFORMATION: {
                TransformationParametersEnum.TRANSFORMATION_TYPE: self.tt_enum.NO_TRANSFORMATION
            }
        }
        transform_params = specific_parameters.get(self.csp_enum.TRANSFORMATION)
        transform_function = self.factory.get_transformation_function(transform_params)
        self.transformed_scores = transform_function(predictions=self.expected_scores[:],
                                                     parameters=transform_params)

    def test_score_no_transformation(self):
        # the scores are not changed (transformation is set to "NO_TRANSFORMATION")
        self.assertListEqual(self.expected_scores, round_list(self.transformed_scores.tolist(), 3))
