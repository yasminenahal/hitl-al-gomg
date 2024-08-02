import unittest

from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.score_components import CustomAlerts
from unittest_reinvent.scoring_tests.scoring_components.fixtures import score_single, score
from reinvent_scoring.scoring.enums import ScoringFunctionComponentNameEnum
from unittest_reinvent.fixtures.test_data import HEXANE, PENTANE, METAMIZOLE, CAFFEINE
import numpy.testing as npt


class TestCustomAlertsWithUserAlerts(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        sf_enum = ScoringFunctionComponentNameEnum()
        list_of_alerts = [CAFFEINE]
        parameters = ComponentParameters(component_type=sf_enum.CUSTOM_ALERTS,
                                         name="custom_alerts",
                                         weight=1.,
                                         specific_parameters={"smiles":list_of_alerts})
        cls.component = CustomAlerts(parameters)

    def test_user_alert_1(self):
        npt.assert_almost_equal(score(self.component, [PENTANE, HEXANE]), [1.0, 1.0])

    def test_user_alert_2(self):
        npt.assert_almost_equal(score_single(self.component, METAMIZOLE), 1.0)
