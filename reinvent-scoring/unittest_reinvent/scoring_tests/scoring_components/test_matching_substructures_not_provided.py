import numpy.testing as npt

from reinvent_scoring import MatchingSubstructure
from unittest_reinvent.fixtures.test_data import METAMIZOLE
from unittest_reinvent.scoring_tests.scoring_components.base_matching_substructure import \
    BaseTestMatchingSubstructure
from unittest_reinvent.scoring_tests.scoring_components.fixtures import score_single


class TestMatchingSubstructuresNotProvided(BaseTestMatchingSubstructure):

    def setUp(self):
        self.smiles = []
        super().setUp()
        self.component = MatchingSubstructure(self.parameters)

    def test_match_no_structure_1(self):
        npt.assert_almost_equal(score_single(self.component, METAMIZOLE), 1.0)
