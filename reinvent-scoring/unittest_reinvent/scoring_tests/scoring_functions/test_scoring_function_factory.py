import unittest

from reinvent_scoring.scoring.enums import ScoringFunctionComponentNameEnum
from reinvent_scoring.scoring.enums import ScoringFunctionNameEnum
from reinvent_scoring.scoring.scoring_function_factory import ScoringFunctionFactory
from reinvent_scoring.scoring.scoring_function_parameters import ScoringFunctionParameters
from unittest_reinvent.fixtures.test_data import BUTANE, CELECOXIB, HEXANE


class TestScoringFunctionFactory(unittest.TestCase):

    def setUp(self):
        enum = ScoringFunctionComponentNameEnum()
        ts_parameters = dict(component_type=enum.TANIMOTO_SIMILARITY,
                             name="tanimoto_similarity",
                             weight=1.,
                             specific_parameters={"smiles":[BUTANE, CELECOXIB]})
        sf_enum = ScoringFunctionNameEnum()
        sf_parameters = ScoringFunctionParameters(name=sf_enum.CUSTOM_SUM, parameters=[ts_parameters])
        self.sf_instance = ScoringFunctionFactory(sf_parameters=sf_parameters)

    def test_sf_factory_1(self):
        result = self.sf_instance.get_final_score([BUTANE])
        self.assertEqual(1., result.total_score)

    def test_sf_factory_2(self):
        result = self.sf_instance.get_final_score([HEXANE])
        self.assertAlmostEqual(result.total_score[0], 0.529, 3)

