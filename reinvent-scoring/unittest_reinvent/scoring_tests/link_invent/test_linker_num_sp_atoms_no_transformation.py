import numpy as np
import numpy.testing as npt

from unittest_reinvent.fixtures.test_data import CELECOXIB_LABELED_PARTS, METAMIZOLE_LABELED_PARTS, \
    AMOXAPINE_LABELED_PARTS, METHOXYHYDRAZINE_LABELED_PARTS, COCAINE_LABELED_PARTS
from unittest_reinvent.scoring_tests.physchem.base_setup import BaseSetup


class TestLinkerNumSPAtomsNoTransformation(BaseSetup):
    def setUp(self):
        super().setup_attrs()
        super().init(self.sf_enum.LINKER_NUM_SP_ATOMS, {})
        super().setUp()

    def test_linker_num_sp_atoms_no_transformation(self):
        smiles = [CELECOXIB_LABELED_PARTS, METAMIZOLE_LABELED_PARTS, AMOXAPINE_LABELED_PARTS,
                  METHOXYHYDRAZINE_LABELED_PARTS, COCAINE_LABELED_PARTS]
        values = np.array([0, 0, 0, 0, 0])
        score = self.sf_state.get_final_score(smiles=smiles)
        npt.assert_array_equal(score.total_score, values)
