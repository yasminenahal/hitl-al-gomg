import numpy as np
import numpy.testing as npt

from reinvent_scoring.scoring.enums import TransformationParametersEnum

from unittest_reinvent.fixtures.test_data import CELECOXIB_LABELED_PARTS, METAMIZOLE_LABELED_PARTS, \
    AMOXAPINE_LABELED_PARTS, METHOXYHYDRAZINE_LABELED_PARTS, COCAINE_LABELED_PARTS
from unittest_reinvent.scoring_tests.physchem.base_setup import BaseSetup


class TestLinkerGraphLengthScoreNoTransformation(BaseSetup):
    def setUp(self):
        super().setup_attrs()
        super().init(self.sf_enum.LINKER_GRAPH_LENGTH, {})
        super().setUp()

    def test_linker_graph_length_no_transformation(self):
        smiles = [CELECOXIB_LABELED_PARTS, METAMIZOLE_LABELED_PARTS, AMOXAPINE_LABELED_PARTS,
                  METHOXYHYDRAZINE_LABELED_PARTS, COCAINE_LABELED_PARTS]
        values = np.array([8., 5., 7., 1., 7.])
        score = self.sf_state.get_final_score(smiles=smiles)
        npt.assert_array_almost_equal(score.total_score, values, 2)
