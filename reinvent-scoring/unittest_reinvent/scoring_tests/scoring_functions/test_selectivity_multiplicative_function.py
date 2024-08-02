from reinvent_scoring.scoring import CustomProduct
from unittest_reinvent.fixtures.test_data import CELECOXIB, BENZENE, ANILINE
from unittest_reinvent.scoring_tests.scoring_functions.base_test_selectivity_multiplicative import \
    BaseTestSelectivityMultiplicative


class TestSelectivityMultiplicativeFunction(BaseTestSelectivityMultiplicative):

    def setUp(self):
        super().init(matching_substructure_smiles=[BENZENE])
        super().setUp()

        self.sf_state = CustomProduct(
            parameters=[self.activity, self.qed_score, self.matching_substructure]
        )

    def test_special_selectivity_multiplicative_1(self):
        score = self.sf_state.get_final_score(smiles=[ANILINE])
        self.assertAlmostEqual(score.total_score[0], 0.269, 3)

    def test_special_selectivity_multiplicative_2(self):
        score_1 = self.sf_state.get_final_score(smiles=[CELECOXIB])
        score_2 = self.sf_state.get_final_score(smiles=[CELECOXIB])
        self.assertAlmostEqual(score_1.total_score[0], 0.337, 3)
        self.assertAlmostEqual(score_2.total_score[0], 0.337, 3)
