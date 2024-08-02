import numpy.testing as npt

from unittest_reinvent.fixtures.test_data import CELECOXIB, ANILINE

from unittest_reinvent.scoring_tests.scoring_components.fixtures import score
from unittest_reinvent.scoring_tests.scoring_components.base_selectivity_component import BaseTestSelectivityComponent
from unittest_reinvent.scoring_tests.fixtures.predictive_model_fixtures import create_activity_component_regression, \
    create_offtarget_activity_component_regression


class TestRegressionSelectivityComponent(BaseTestSelectivityComponent):
    def setUp(self):
        self.activity = create_activity_component_regression()
        self.off_activity = create_offtarget_activity_component_regression()
        super().setUp()

    def test_selectivity_component(self):
        smiles = [CELECOXIB, ANILINE]
        expected_values = [0.053, 0.053]
        scores = score(self.component, smiles)
        npt.assert_almost_equal(scores, expected_values, decimal=3)

