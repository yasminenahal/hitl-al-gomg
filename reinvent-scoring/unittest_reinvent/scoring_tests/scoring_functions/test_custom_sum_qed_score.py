from rdkit import Chem

from reinvent_scoring.scoring.score_components.score_component_factory import ScoreComponentFactory
from unittest_reinvent.fixtures.test_data import BENZENE
from unittest_reinvent.scoring_tests.scoring_functions.base_test_custom_sum import BaseTestCustomSum


class Test_custom_sum_qed_score(BaseTestCustomSum):

    def setUp(self):
        self.setup_attrs()
        super().init(self.sf_enum.QED_SCORE, "qed_score")
        super().setUp()

    def test_qed_sum(self):
        score = self.sf_state.get_final_score(smiles=[BENZENE])
        self.assertAlmostEqual(score.total_score[0], 0.4426, 4)

    def test_qed_component_factory(self):
        factory = ScoreComponentFactory([self.parameters])
        scoring_components = factory.create_score_components()
        sc = scoring_components[0]
        smile = BENZENE
        mol = Chem.MolFromSmiles(smile)
        score = sc.calculate_score([mol])
        self.assertAlmostEqual(score.total_score[0], 0.4426, 4)
