from reinvent_scoring.scoring.enums import TransformationParametersEnum
from unittest_reinvent.scoring_tests.scoring_transformations.base_test_score_transformations import BaseTestScoreTransformation


class TestScoreRightStepTransformation(BaseTestScoreTransformation):

    def setUp(self):
        super().setup_attrs()
        specific_parameters = {
            self.csp_enum.TRANSFORMATION: {
                TransformationParametersEnum.TRANSFORMATION_TYPE: self.tt_enum.RIGHT_STEP
            }
        }
        super().init(specific_parameters)
        super().setUp()

    def test_right_step_transformation(self):
        self.assertListEqual([1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                             self.transformed_scores.tolist())

    def test_right_step_transformation_updated_parameters(self):
        self.update_parameters({
            self.csp_enum.TRANSFORMATION: {
                TransformationParametersEnum.LOW: 25
            }
        })
        self.assertListEqual([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
                             self.transformed_scores.tolist())
