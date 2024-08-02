import numpy as np
import numpy.testing as npt

from unittest_reinvent.fixtures.test_data import CELECOXIB_LABELED_PARTS, METAMIZOLE_LABELED_PARTS, \
    AMOXAPINE_LABELED_PARTS, METHOXYHYDRAZINE_LABELED_PARTS, COCAINE_LABELED_PARTS
from unittest_reinvent.scoring_tests.physchem.base_setup import BaseSetup


class TestLinkerLengthRatioScoreNoTransformation(BaseSetup):
    def setUp(self):
        super().setup_attrs()
        super().init(self.sf_enum.LINKER_LENGTH_RATIO, {})
        super().setUp()

    def test_linker_length_ratio_no_transformation(self):
        smiles = [CELECOXIB_LABELED_PARTS, METAMIZOLE_LABELED_PARTS, AMOXAPINE_LABELED_PARTS,
                  METHOXYHYDRAZINE_LABELED_PARTS, COCAINE_LABELED_PARTS]
        values = np.array([25, 80, 42.86, 100, 42.86])
        score = self.sf_state.get_final_score(smiles=smiles)
        npt.assert_array_almost_equal(score.total_score, values, 2)
