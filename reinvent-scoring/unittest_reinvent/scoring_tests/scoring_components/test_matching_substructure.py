import numpy.testing as npt

from reinvent_scoring.scoring.score_components import MatchingSubstructure
from unittest_reinvent.scoring_tests.scoring_components.fixtures import score_single
from unittest_reinvent.fixtures.test_data import COCAINE, CAFFEINE, CELECOXIB
from unittest_reinvent.scoring_tests.scoring_components.base_matching_substructure import \
    BaseTestMatchingSubstructure


class TestMatchingSubstructures(BaseTestMatchingSubstructure):

    def setUp(self):
        self.smiles = [COCAINE]
        super().setUp()
        self.component = MatchingSubstructure(self.parameters)

    def test_match_1(self):
        npt.assert_almost_equal(score_single(self.component, CAFFEINE), 0.5)

    def test_match_2(self):
        npt.assert_almost_equal(score_single(self.component, CELECOXIB), 0.5)
