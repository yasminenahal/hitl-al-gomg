import unittest

from reinvent_scoring import ComponentParameters, CustomSum
from reinvent_scoring.scoring.enums import ScoringFunctionComponentNameEnum
from unittest_reinvent.fixtures.test_data import INVALID, PROPANE


class TestInvalidMatchingSubstructure(unittest.TestCase):

    def setUp(self):
        sf_enum = ScoringFunctionComponentNameEnum()
        self.matching_pattern = INVALID
        self.ts_parameters = ComponentParameters(component_type=sf_enum.MATCHING_SUBSTRUCTURE,
                                                 name="matching_substructure",
                                                 weight=1.,
                                                 specific_parameters={"smiles":[self.matching_pattern]})

    def test_match_invalid_structure_1(self):
        with self.assertRaises(IOError) as context:
            self.sf_state = CustomSum(parameters=[self.ts_parameters])
            self.sf_state.get_final_score(smiles=[PROPANE])
        msg = f"Invalid smarts pattern provided as a matching substructure: {self.matching_pattern}"
        self.assertEqual(msg, str(context.exception))
