import numpy as np
import numpy.testing as npt

from reinvent_scoring.scoring.enums import TransformationParametersEnum

from unittest_reinvent.fixtures.test_data import CELECOXIB_LABELED_PARTS, METAMIZOLE_LABELED_PARTS, \
    AMOXAPINE_LABELED_PARTS, METHOXYHYDRAZINE_LABELED_PARTS, COCAINE_LABELED_PARTS
from unittest_reinvent.scoring_tests.physchem.base_setup import BaseSetup


class TestLinkerNumHBAWithDoubleSigmoid(BaseSetup):
    def setUp(self):
        super().setup_attrs()
        specific_parameters = {
            self.csp_enum.TRANSFORMATION: {
                TransformationParametersEnum.LOW: 1,
                TransformationParametersEnum.HIGH: 4,
                TransformationParametersEnum.COEF_DIV: 10,
                TransformationParametersEnum.COEF_SI: 150,
                TransformationParametersEnum.COEF_SE: 150,
                TransformationParametersEnum.TRANSFORMATION_TYPE: self.tt_enum.DOUBLE_SIGMOID
            }
        }
        super().init(self.sf_enum.LINKER_NUM_HBA, specific_parameters)
        super().setUp()

    def test_linker_num_hba_with_double_sigmoid(self):
        smiles = [CELECOXIB_LABELED_PARTS, METAMIZOLE_LABELED_PARTS, AMOXAPINE_LABELED_PARTS,
                  METHOXYHYDRAZINE_LABELED_PARTS, COCAINE_LABELED_PARTS]
        values = np.array([0.5, 1., 1., 1., 1.])
        score = self.sf_state.get_final_score(smiles=smiles)
        npt.assert_array_almost_equal(score.total_score, values, 2)
