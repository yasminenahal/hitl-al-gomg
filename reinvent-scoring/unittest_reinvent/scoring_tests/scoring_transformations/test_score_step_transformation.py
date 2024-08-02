from reinvent_scoring.scoring.enums import TransformationParametersEnum
from unittest_reinvent.scoring_tests.scoring_transformations.base_test_score_transformations import BaseTestScoreTransformation


class TestScoreStepTransformation(BaseTestScoreTransformation):

    def setUp(self):
        super().setup_attrs()
        specific_parameters = {
            self.csp_enum.TRANSFORMATION: {
                TransformationParametersEnum.HIGH: 14,
                TransformationParametersEnum.TRANSFORMATION_TYPE: self.tt_enum.STEP
            }
        }
        super().init(specific_parameters)
        super().setUp()

    def test_step_transformation(self):
        self.assertListEqual([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0], self.transformed_scores.tolist())
