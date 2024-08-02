import unittest

from unittest_reinvent.fixtures.test_data import CAFFEINE, ANILINE, TOLUENE, CELECOXIB_C
from unittest_reinvent.scoring_tests.scoring_components.fixtures import instantiate_component, score_single
import numpy.testing as npt


class TestTanimotoCustomFingerprintParameters(unittest.TestCase):

    def test_default(self):
        component = instantiate_component(specific_parameters={})
        npt.assert_almost_equal(score_single(component, CAFFEINE), 0.101, decimal=3)

    def test_custom_radius(self):
        component = instantiate_component(specific_parameters={"radius": 2})
        npt.assert_almost_equal(score_single(component, CAFFEINE), 0.125, decimal=3)

    def test_custom_use_counts(self):
        component = instantiate_component(specific_parameters={"use_counts": False})
        npt.assert_almost_equal(score_single(component, CAFFEINE), 0.054, decimal=3)

    def test_custom_use_features_1(self):
        component = instantiate_component(specific_parameters={"use_features": False})
        npt.assert_almost_equal(score_single(component, CAFFEINE), 0.083, decimal=3)

    def test_custom_use_features_2(self):
        smiles = [ANILINE]
        component = instantiate_component(
            smiles=smiles, specific_parameters={"use_features": False})
        npt.assert_almost_equal(score_single(component, TOLUENE), 0.517, decimal=3)

    def test_custom_use_features_3(self):
        component = instantiate_component(specific_parameters={"use_features": False})
        npt.assert_almost_equal(score_single(component, CELECOXIB_C), 0.810, decimal=3)
