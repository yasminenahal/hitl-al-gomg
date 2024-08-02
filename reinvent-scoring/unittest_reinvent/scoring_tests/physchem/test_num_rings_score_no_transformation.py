import numpy as np
import numpy.testing as npt

from unittest_reinvent.fixtures.test_data import BENZENE, ANILINE, BENZO_A_PYRENE, DECALIN, PACLITAXEL
from unittest_reinvent.scoring_tests.physchem.base_setup import BaseSetup


class TestNumRingsScoreNoTransformation(BaseSetup):

    def setUp(self):
        super().setup_attrs()
        super().init(self.sf_enum.NUM_RINGS, {})
        super().setUp()

    def test_num_rings_1(self):
        smiles = [BENZENE, ANILINE, BENZO_A_PYRENE, DECALIN, PACLITAXEL]
        values = np.array([1., 1., 5., 2., 7.])
        score = self.sf_state.get_final_score(smiles=smiles)
        npt.assert_array_almost_equal(score.total_score, values, 2)
