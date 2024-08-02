from unittest_reinvent.scoring_tests.fixtures.predictive_model_fixtures import create_activity_component_regression, \
    create_offtarget_activity_component_classification

from unittest_reinvent.scoring_tests.scoring_components.fixtures import score
from unittest_reinvent.fixtures.test_data import CELECOXIB, ASPIRIN

import numpy.testing as npt
from unittest_reinvent.scoring_tests.scoring_components.base_selectivity_component import BaseTestSelectivityComponent


class TestMixedSelectivityComponent(BaseTestSelectivityComponent):
    def setUp(self):
        self.activity = create_activity_component_regression()
        self.off_activity = create_offtarget_activity_component_classification()
        super().setUp()

    def test_selectivity_component(self):
        smiles = [CELECOXIB, ASPIRIN]
        expected_values = [0.01, 0.01]
        scores = score(self.component, smiles)
        npt.assert_almost_equal(scores, expected_values, decimal=3)


