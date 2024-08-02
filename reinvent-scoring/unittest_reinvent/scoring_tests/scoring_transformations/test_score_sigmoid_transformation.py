from reinvent_scoring.scoring.enums import TransformationParametersEnum
from unittest_reinvent.scoring_tests.scoring_transformations.base_test_score_sigmoid_transformations import \
    BaseTestSigmoidScoreTransformation
from unittest_reinvent.scoring_tests.scoring_transformations.fixtures import round_list


class TestSigmoidTransformations(BaseTestSigmoidScoreTransformation):

    def setUp(self):
        super().setup_attrs()
        specific_parameters = {
            self.csp_enum.TRANSFORMATION: {
                TransformationParametersEnum.TRANSFORMATION_TYPE: self.tt_enum.SIGMOID
            }
        }
        super().init(specific_parameters)
        super().setUp()

    def test_sigmoid_transformation(self):
        self.assertListEqual(
            [0.998, 0.001, 0.834, 1.0, 1.0, 0.053, 1.0],
            round_list(self.transformed_scores.tolist(), 3))

    def test_sigmoid_transformation_updated_parameters(self):
        self.update_parameters({
            self.csp_enum.TRANSFORMATION: {
                TransformationParametersEnum.HIGH: 45,
                TransformationParametersEnum.LOW: 9
            }
        })
        self.assertListEqual(
            [0.083, 0.013, 0.045, 1, 1, 0.025, 1],
            round_list(self.transformed_scores.tolist(), 3))

