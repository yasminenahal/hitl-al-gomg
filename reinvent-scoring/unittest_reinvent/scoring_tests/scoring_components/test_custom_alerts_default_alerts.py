import unittest

from reinvent_scoring.scoring.score_components import CustomAlerts
from unittest_reinvent.scoring_tests.fixtures.predictive_model_fixtures import create_custom_alerts_configuration
from unittest_reinvent.scoring_tests.scoring_components.fixtures import score_single, score
from unittest_reinvent.fixtures.test_data import HEXANE, PENTANE, METAMIZOLE, CAFFEINE
import numpy.testing as npt


class TestCustomAlertsWithDefaultAlerts(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        parameters = create_custom_alerts_configuration()
        cls.component = CustomAlerts(parameters)

    def test_alert_1(self):
        npt.assert_almost_equal(score(self.component, [PENTANE, HEXANE]), [1, 1])

    def test_alert_2(self):
        npt.assert_almost_equal(score_single(self.component, METAMIZOLE), 0)
