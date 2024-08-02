from reinvent_scoring import ComponentParameters, CustomProduct
from reinvent_scoring.scoring.enums import TransformationParametersEnum
from unittest_reinvent.fixtures.paths import ACTIVITY_REGRESSION
from unittest_reinvent.fixtures.test_data import GENTAMICIN, ASPIRIN, ANILINE, CELECOXIB
from unittest_reinvent.scoring_tests.fixtures import create_offtarget_activity_component_regression
from unittest_reinvent.scoring_tests.scoring_functions.base_test_selectivity_double_sigmoid import \
    BaseTestSelectivityFunctionDoubleSigmoid


class TestSpecialDesirabilityMultiplicativeFunction(BaseTestSelectivityFunctionDoubleSigmoid):

    def setUp(self):
        smiles = [GENTAMICIN]
        super().init(matching_substructure_smiles=smiles)
        super().setUp()

        self.off_activity.weight = 4
        self.off_activity.specific_parameters[self.csp_enum.TRANSFORMATION] = {
            TransformationParametersEnum.TRANSFORMATION_TYPE: self.transf_type.DOUBLE_SIGMOID,
            TransformationParametersEnum.COEF_DIV: 100.,
            TransformationParametersEnum.COEF_SI: 150.,
            TransformationParametersEnum.COEF_SE: 150.,
        }
        self.off_activity.specific_parameters[self.csp_enum.MODEL_PATH] = ACTIVITY_REGRESSION
        self.off_activity2 = create_offtarget_activity_component_regression()
        self.off_activity2.weight = 4

        selectivity_1 = ComponentParameters(component_type=self.sf_enum.SELECTIVITY,
                                            name="selectivity_1",
                                            weight=4.,
                                            specific_parameters={
                                                "activity_model_path": self.activity.specific_parameters.get(self.csp_enum.MODEL_PATH),
                                                "offtarget_model_path": self.off_activity.specific_parameters.get(self.csp_enum.MODEL_PATH),
                                                "activity_specific_parameters": self.activity.specific_parameters.copy(),
                                                "offtarget_specific_parameters": self.off_activity.specific_parameters,
                                                "delta_transformation_parameters": self.delta_params
                                            })

        selectivity_2 = ComponentParameters(component_type=self.sf_enum.SELECTIVITY,
                                            name="selectivity_2",
                                            weight=4.,
                                            specific_parameters={
                                                "activity_model_path": self.activity.specific_parameters.get(self.csp_enum.MODEL_PATH),
                                                "offtarget_model_path": self.off_activity2.specific_parameters.get(self.csp_enum.MODEL_PATH),
                                                "activity_specific_parameters": self.activity.specific_parameters.copy(),
                                                "offtarget_specific_parameters": self.off_activity2.specific_parameters,
                                                "delta_transformation_parameters": self.delta_params
                                            })

        self.sf_state = CustomProduct(
            parameters=[self.activity, selectivity_1, selectivity_2, self.qed_score, self.custom_alerts,
                        self.matching_substructure])

    def test_special_desirability_multiplicative_1(self):
        score = self.sf_state.get_final_score(smiles=[ASPIRIN])
        self.assertAlmostEqual(score.total_score[0], 0.0451, 3)

    def test_special_desirability_multiplicative_2(self):
        score = self.sf_state.get_final_score(smiles=[ANILINE])
        self.assertAlmostEqual(score.total_score[0], 0.044, 3)

    def test_special_desirability_multiplicative_3(self):
        score = self.sf_state.get_final_score(smiles=[CELECOXIB])
        self.assertAlmostEqual(score.total_score[0], 0.046, 3)
