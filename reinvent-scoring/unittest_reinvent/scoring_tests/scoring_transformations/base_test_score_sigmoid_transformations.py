from reinvent_scoring import TransformationFactory
from reinvent_scoring.scoring.enums import TransformationParametersEnum
from unittest_reinvent.scoring_tests.scoring_transformations.base_test_score_transformations import BaseTestScoreTransformation


class BaseTestSigmoidScoreTransformation(BaseTestScoreTransformation):

    def setUp(self):
        self.v_list = [12.016, 0.0015, 7.9, 123.264, 77.80, 4.0, 111.12]
        self.factory = TransformationFactory()
        base_specific_parameters = {
            self.csp_enum.TRANSFORMATION: {
                TransformationParametersEnum.LOW: 4,
                TransformationParametersEnum.HIGH: 9,
                TransformationParametersEnum.K: 0.25
            }
        }

        self.specific_parameters = self._merge_parameters(
            base_specific_parameters, self.added_specific_parameters)

        transform_params = self.specific_parameters.get(self.csp_enum.TRANSFORMATION)
        transform_function = self.factory.get_transformation_function(transform_params)
        self.transformed_scores = transform_function(predictions=self.v_list[:],
                                                     parameters=transform_params)
