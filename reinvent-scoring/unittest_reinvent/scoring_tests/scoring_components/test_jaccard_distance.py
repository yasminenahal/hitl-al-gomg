import unittest

from unittest_reinvent.scoring_tests.scoring_components.fixtures import score_single, score, \
    instantiate_jaccard_component
from unittest_reinvent.fixtures.test_data import CELECOXIB, CELECOXIB_C, BUTANE
import numpy.testing as npt


class TestJaccardDistance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.component = instantiate_jaccard_component()

    def test_distance_1(self):
        npt.assert_almost_equal(score_single(self.component, BUTANE), 0.0)

    def test_distance_2(self):
        npt.assert_almost_equal(score_single(self.component, CELECOXIB), 0.0)

    def test_distance_3(self):
        npt.assert_almost_equal(score_single(self.component, CELECOXIB_C), 0.109, 3)

    def test_distance_4(self):
        smiles = [CELECOXIB, BUTANE]
        npt.assert_almost_equal(score(self.component, smiles), [0, 0])
