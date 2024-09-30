import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem, rdFingerprintGenerator


class ecfp_generator:
    def __init__(self, bitSize: int = 2048, radius: int = 2, useCounts: bool = False):
        self.bitSize = bitSize
        self.radius = radius
        self.fpgen = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=bitSize
        )
        self.generate = (
            self.fpgen.GetCountFingerprintAsNumPy
            if useCounts
            else self.fpgen.GetFingerprintAsNumPy
        )

    def get_fingerprints(self, smiles) -> np.ndarray:
        """
        The function `get_fingerprints` takes a list of SMILES strings as input and returns a numpy array of
        fingerprints.
        Args:
            smiles (List[str]): A list of SMILES strings representing chemical compounds
        Returns:
            np.ndarray[np.int32]: an array of fingerprints, where each fingerprint is represented as an array of integers.
        """
        fps = np.stack([self.generate(Chem.MolFromSmiles(smile)) for smile in smiles])
        return fps


def double_sigmoid(x, low, high, alpha_1, alpha_2):
    return 10 ** (x * alpha_1) / (10 ** (x * alpha_1) + 10 ** (low * alpha_1)) - 10 ** (
        x * alpha_2
    ) / (10 ** (x * alpha_2) + 10 ** (high * alpha_2))


def get_mol(smiles_or_mol):
    """
    Loads SMILES/molecule into RDKit's object
    """
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    return smiles_or_mol


def canonic_smiles(smiles_or_mol):
    mol = get_mol(smiles_or_mol)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)
