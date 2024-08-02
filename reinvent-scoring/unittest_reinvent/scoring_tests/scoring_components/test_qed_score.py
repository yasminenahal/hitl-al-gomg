import unittest

from rdkit import Chem

from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.score_components import QedScore
from unittest_reinvent.scoring_tests.scoring_components.fixtures import score_single
from reinvent_scoring.scoring.enums import ScoringFunctionComponentNameEnum
from unittest_reinvent.fixtures.test_data import HEXANE
import numpy.testing as npt


class TestQedScore(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        sf_enum = ScoringFunctionComponentNameEnum()
        parameters = ComponentParameters(component_type=sf_enum.QED_SCORE,
                                         name="qed_score",
                                         weight=1.,
                                         specific_parameters={})
        cls.component = QedScore(parameters)
        cls.smile = HEXANE
        cls.mol = Chem.MolFromSmiles(cls.smile)

    def test_molecule_parsed_successfully(self):
        self.assertIsNotNone(self.mol)

    def test_invalid_molecule_returns_zero(self):
        score = self.component.calculate_score([None])
        npt.assert_almost_equal(score.total_score[0], 0.0, 4)

    def test_one_molecule(self):
        score = self.component.calculate_score([self.mol])
        self.assertEqual(1, len(score.total_score))
        npt.assert_almost_equal(score.total_score[0], 0.463, 4)

    def test_one_molecule_2(self):
        npt.assert_almost_equal(score_single(self.component, self.smile), 0.463, 3)

    def test_two_molecules(self):
        score = self.component.calculate_score([self.mol, self.mol])
        self.assertEqual(2, len(score.total_score))
        npt.assert_almost_equal(score.total_score[0], 0.463, 4)
        npt.assert_almost_equal(score.total_score[1], 0.463, 4)
