from reinvent_scoring import CustomProduct, FinalSummary
from unittest_reinvent.fixtures.test_data import AMOXAPINE, ASPIRIN, CELECOXIB
from unittest_reinvent.scoring_tests.scoring_functions.base_test_selectivity_double_sigmoid import \
    BaseTestSelectivityFunctionDoubleSigmoid


class TestSelectivityFunctionWithDoubleSigmoid(BaseTestSelectivityFunctionDoubleSigmoid):

    def setUp(self):
        smiles = [AMOXAPINE]
        super().init(matching_substructure_smiles=smiles)
        super().setUp()

        self.sf_state = CustomProduct(
            parameters=[self.activity, self.selectivity, self.qed_score, self.matching_substructure,
                        self.custom_alerts])

    def test_selectivity_function_with_scikit_and_wrapped_models_1(self):
        score: FinalSummary = self.sf_state.get_final_score(smiles=[ASPIRIN])
        self.assertAlmostEqual(score.total_score[0], 0.153, 3)

    def test_selectivity_function_with_scikit_and_wrapped_models_2(self):
        score = self.sf_state.get_final_score(smiles=[CELECOXIB])
        self.assertAlmostEqual(score.total_score[0], 0.169, 3)
