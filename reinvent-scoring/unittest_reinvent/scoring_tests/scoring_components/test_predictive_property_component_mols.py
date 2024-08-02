from typing import Union, List

import numpy.testing as npt
import rdkit
from rdkit.Chem.rdmolfiles import MolFromSmiles

from unittest_reinvent.scoring_tests.scoring_components.fixtures import score_single, score
from unittest_reinvent.fixtures.test_data import CELECOXIB, BUTANE, PENTANE
from unittest_reinvent.scoring_tests.scoring_components.base_predictive_property_component import \
    BaseTestPredictivePropertyComponent


class ModelWithPredictFromMols:

    @staticmethod
    def predict_from_rdkit_mols(mols: List[rdkit.Chem.Mol]) -> List[float]:
        output = [mol.GetNumAtoms() for mol in mols]
        return output

    @staticmethod
    def predict_from_smiles(smiles: Union[str, List[str]]) -> List[float]:
        mols = [MolFromSmiles(s) for s in smiles]
        output = [mol.GetNumAtoms() for mol in mols]
        return output


class TestPredictivePropertyComponentWithPredictFromMols(BaseTestPredictivePropertyComponent):

    @classmethod
    def setUpClass(cls):
        cls.model = ModelWithPredictFromMols()
        super().setUpClass()

    def test_predictive_property_1(self):
        npt.assert_almost_equal(score_single(self.component, CELECOXIB), 26, 3)

    def test_predictive_property_2(self):
        npt.assert_almost_equal(score(self.component, [BUTANE, PENTANE]), [4, 5], 3)
