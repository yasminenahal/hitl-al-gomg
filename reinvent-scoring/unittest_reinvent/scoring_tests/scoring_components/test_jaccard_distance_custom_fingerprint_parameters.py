import unittest

from unittest_reinvent.scoring_tests.scoring_components.fixtures import score_single, instantiate_jaccard_component
from unittest_reinvent.fixtures.test_data import CELECOXIB_C, CAFFEINE, ANILINE, AMOXAPINE
import numpy.testing as npt


class TestJaccardCustomFingerprintParameters(unittest.TestCase):

    def test_default(self):
        component = instantiate_jaccard_component(specific_parameters={})
        npt.assert_almost_equal(score_single(component, CAFFEINE), 0.898, decimal=3)

    def test_custom_radius(self):
        component = instantiate_jaccard_component(specific_parameters={"radius": 2})
        npt.assert_almost_equal(score_single(component, CAFFEINE), 0.875, decimal=3)

    def test_custom_use_counts(self):
        component = instantiate_jaccard_component(specific_parameters={"use_counts": False})
        npt.assert_almost_equal(score_single(component, CAFFEINE), 0.945, decimal=3)

    def test_custom_use_features_1(self):
        component = instantiate_jaccard_component(specific_parameters={"use_features": False})
        npt.assert_almost_equal(score_single(component, CAFFEINE), 0.916, decimal=3)

    def test_custom_use_features_2(self):
        smiles = [ANILINE]
        component = instantiate_jaccard_component(
            smiles=smiles, specific_parameters={"use_features": False}
        )
        npt.assert_almost_equal(score_single(component, AMOXAPINE), 0.872, decimal=3)

    def test_custom_use_features_3(self):
        component = instantiate_jaccard_component(specific_parameters={"use_features": False})
        npt.assert_almost_equal(score_single(component, CELECOXIB_C), 0.189, decimal=3)
