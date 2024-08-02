import unittest

from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring import CustomProduct
from reinvent_scoring.scoring.enums import ScoringFunctionComponentNameEnum
from unittest_reinvent.fixtures.test_data import CELECOXIB, BENZENE, PENTANE


class TestPrimaryMultWithNoActivity(unittest.TestCase):

    def setUp(self):
        enum = ScoringFunctionComponentNameEnum()
        custom_alerts = ComponentParameters(component_type=enum.CUSTOM_ALERTS,
                                            name="custom_alerts_name",
                                            weight=1.,
                                            specific_parameters={"smiles":[PENTANE]})
        matching_substructure = ComponentParameters(component_type=enum.MATCHING_SUBSTRUCTURE,
                                                    name="matching_substructure_name",
                                                    weight=1.,
                                                    specific_parameters={"smiles":[BENZENE]})
        self.sf_state = CustomProduct(
            parameters=[matching_substructure, custom_alerts])

    def test_primary_mult_with_alert_match_1(self):
        score = self.sf_state.get_final_score(smiles=[PENTANE])
        self.assertEqual(score.total_score, 0)

    def test_primary_mult_with_alert_match_2(self):
        score = self.sf_state.get_final_score(smiles=[CELECOXIB])
        self.assertAlmostEqual(score.total_score[0], 1, 3)
