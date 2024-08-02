import numpy.testing as npt

from unittest_reinvent.fixtures.test_data import CELECOXIB
from unittest_reinvent.scoring_tests.fixtures import create_activity_component_classification, \
    create_offtarget_activity_component_classification
from unittest_reinvent.scoring_tests.scoring_components.fixtures import score_single
from unittest_reinvent.scoring_tests.scoring_components.base_selectivity_component import BaseTestSelectivityComponent


class TestClassificationSelectivityComponent(BaseTestSelectivityComponent):

    def setUp(self):
        self.activity = create_activity_component_classification()
        self.off_activity = create_offtarget_activity_component_classification()
        super().setUp()

    def test_selectivity_component_1(self):
        npt.assert_almost_equal(score_single(self.component, CELECOXIB), 0.01)
