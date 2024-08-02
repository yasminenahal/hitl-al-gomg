import unittest

from reinvent_scoring.scoring import CustomSum
from unittest_reinvent.scoring_tests.fixtures.predictive_model_fixtures import create_custom_alerts_configuration
from unittest_reinvent.fixtures.test_data import CELECOXIB, CYCLODECANE, CYCLOUNDERCANE


class TestCustomAlertsWithDefaultAlerts(unittest.TestCase):

    def setUp(self):
        parameters = create_custom_alerts_configuration()
        self.sf_state = CustomSum(parameters=[parameters])

    def test_alert_1(self):
        score = self.sf_state.get_final_score(smiles=[CYCLODECANE, CYCLOUNDERCANE])
        for i, s in enumerate(score.total_score):
            with self.subTest(i=i):
                self.assertEqual(score.total_score[i], 0.)

    def test_alert_2(self):
        score = self.sf_state.get_final_score(smiles=[CELECOXIB])
        for i, s in enumerate(score.total_score):
            with self.subTest(i=i):
                self.assertEqual(score.total_score[i], 1.)
