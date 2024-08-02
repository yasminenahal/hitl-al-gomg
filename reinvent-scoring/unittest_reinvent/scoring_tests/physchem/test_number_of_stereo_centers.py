import numpy as np
import numpy.testing as npt

from reinvent_scoring.scoring.enums import TransformationParametersEnum

from unittest_reinvent.fixtures.test_data import BENZENE, ISOHEPTANE
from unittest_reinvent.scoring_tests.physchem.base_setup import BaseSetup


class TestNumberOfStereoCenters(BaseSetup):

    def setUp(self):
        super().setup_attrs()
        specific_parameters = {
            self.csp_enum.TRANSFORMATION: {
                TransformationParametersEnum.TRANSFORMATION_TYPE: self.tt_enum.NO_TRANSFORMATION
            }
        }
        super().init(self.sf_enum.NUMBER_OF_STEREO_CENTERS, specific_parameters)
        super().setUp()

    def test_tpsa_1(self):
        smiles = [ISOHEPTANE, BENZENE]
        values = np.array([1., 0.])
        score = self.sf_state.get_final_score(smiles=smiles)
        npt.assert_array_almost_equal(score.total_score, values, 2)
