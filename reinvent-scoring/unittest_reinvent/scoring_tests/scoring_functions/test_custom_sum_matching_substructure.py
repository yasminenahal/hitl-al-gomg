from unittest_reinvent.fixtures.test_data import PROPANE, CELECOXIB
from unittest_reinvent.scoring_tests.scoring_functions.base_test_custom_sum import BaseTestCustomSum


class TestMatchingSubstructures(BaseTestCustomSum):

    def setUp(self):
        self.setup_attrs()
        smiles = [PROPANE]
        super().init(self.sf_enum.MATCHING_SUBSTRUCTURE, "matching_substructure", smiles)
        super().setUp()

    def _assert_score(self, score, expected_score):
        for i, s in enumerate(score.total_score):
            self.assertEqual(score.total_score[i], expected_score)

    def test_match_1(self):
        score = self.sf_state.get_final_score(smiles=[PROPANE])
        self._assert_score(score, 1)

    def test_match_2(self):
        score = self.sf_state.get_final_score(smiles=[CELECOXIB])
        self._assert_score(score, 0.5)
