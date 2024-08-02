from reinvent_scoring.scoring.enums import TransformationParametersEnum
from unittest_reinvent.scoring_tests.scoring_transformations.base_test_score_sigmoid_transformations import \
    BaseTestSigmoidScoreTransformation
from unittest_reinvent.scoring_tests.scoring_transformations.fixtures import round_list


class TestDoubleSigmoidTransformations(BaseTestSigmoidScoreTransformation):

    def setUp(self):
        super().setup_attrs()
        specific_parameters = {
            self.csp_enum.TRANSFORMATION: {
                TransformationParametersEnum.COEF_DIV: 100,
                TransformationParametersEnum.COEF_SI: 150,
                TransformationParametersEnum.COEF_SE: 150,
                TransformationParametersEnum.TRANSFORMATION_TYPE: self.tt_enum.DOUBLE_SIGMOID
            }
        }
        super().init(specific_parameters)
        super().setUp()

    def test_double_sigmoid_transformation(self):
        self.assertListEqual(
            [0, 0, 0.978, 0.0, 0.0, 0.5, 0.0],
            round_list(self.transformed_scores.tolist(), 3))

    def test_double_sigmoid_transformation_updated_parameters(self):
        self.update_parameters({
            self.csp_enum.TRANSFORMATION: {
                TransformationParametersEnum.COEF_DIV: 200,
                TransformationParametersEnum.COEF_SI: 100,
                TransformationParametersEnum.COEF_SE: 100
            }
        })
        transform_params = self.specific_parameters.get(self.csp_enum.TRANSFORMATION)
        transform_function = self.factory.get_transformation_function(transform_params)
        self.transformed_scores = transform_function(predictions=self.v_list[:],
                                                     parameters=transform_params)
        self.assertListEqual(
            [0.03, 0.01, 0.769, 0.0, 0.0, 0.497, 0.0],
            round_list(self.transformed_scores.tolist(), 3))
