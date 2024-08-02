from unittest_reinvent.fixtures.test_data import PROPANE
from unittest_reinvent.scoring_tests.scoring_functions.base_test_custom_sum import BaseTestCustomSum


class TestMatchingSubstructuresNotProvided(BaseTestCustomSum):

    def setUp(self):
        super().setup_attrs()
        super().init(self.sf_enum.MATCHING_SUBSTRUCTURE, "matching_substructure")
        super().setUp()

    def test_match_no_structure_1(self):
        score = self.sf_state.get_final_score(smiles=[PROPANE])
        for i, s in enumerate(score.total_score):
            self.assertEqual(score.total_score[i], 1.)
