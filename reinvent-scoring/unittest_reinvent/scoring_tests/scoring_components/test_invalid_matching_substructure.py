from reinvent_scoring.scoring.score_components import MatchingSubstructure
from unittest_reinvent.fixtures.test_data import INVALID
from unittest_reinvent.scoring_tests.scoring_components.base_matching_substructure import \
    BaseTestMatchingSubstructure


class TestInvalidMatchingSubstructure(BaseTestMatchingSubstructure):

    def setUp(self):
        self.smiles = [INVALID]
        super().setUp()

    def test_match_invalid_structure_1(self):
        with self.assertRaises(IOError) as context:
            _ = MatchingSubstructure(self.parameters)
        msg = f"Invalid smarts pattern provided as a matching substructure: {INVALID}"
        self.assertEqual(msg, str(context.exception))
