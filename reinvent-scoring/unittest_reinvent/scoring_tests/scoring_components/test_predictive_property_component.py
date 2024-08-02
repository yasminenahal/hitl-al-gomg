import unittest


import numpy.testing as npt

from reinvent_scoring.scoring.enums import ComponentSpecificParametersEnum
from reinvent_scoring.scoring.score_components import PredictivePropertyComponent
from unittest_reinvent.scoring_tests.fixtures.predictive_model_fixtures import \
    create_predictive_property_component_regression
from unittest_reinvent.scoring_tests.scoring_components.fixtures import score_single
from unittest_reinvent.fixtures.test_data import CELECOXIB


class TestPredictivePropertyComponent(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.csp_enum = ComponentSpecificParametersEnum()
        activity = create_predictive_property_component_regression()
        cls.component = PredictivePropertyComponent(activity)

    def test_predictive_property_1(self):
        npt.assert_almost_equal(score_single(self.component, CELECOXIB), 0.15, 3)

    def test_predictive_property_2(self):
        self.assertTrue(self.component.parameters.specific_parameters[self.csp_enum.TRANSFORMATION])


