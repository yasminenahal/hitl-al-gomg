import unittest

import numpy.testing as npt

from unittest_reinvent.fixtures.test_data import CELECOXIB, CELECOXIB_C, BUTANE
from unittest_reinvent.scoring_tests.scoring_components.fixtures import score_single, score, instantiate_component


class TestTanimotoSimilarity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.component = instantiate_component()

    def test_similarity_1(self):
        npt.assert_almost_equal(score_single(self.component, BUTANE), 1.0)

    def test_similarity_2(self):
        npt.assert_almost_equal(score_single(self.component, CELECOXIB), 1.0)

    def test_similarity_3(self):
        npt.assert_almost_equal(score_single(self.component, CELECOXIB_C), 0.89, decimal=3)

    def test_similarity_4(self):
        smiles = [BUTANE, CELECOXIB]
        scores = score(self.component, smiles)
        npt.assert_almost_equal(scores, 1.0)
