from unittest_reinvent.fixtures.test_data import CELECOXIB, ASPIRIN, BENZENE, ANILINE
from unittest_reinvent.scoring_tests.scoring_functions.base_test_custom_sum import BaseTestCustomSum


class TestCustomAlertsWithUserAlerts(BaseTestCustomSum):

    def setUp(self):
        self.setup_attrs()
        smiles = [ASPIRIN]
        super().init(self.sf_enum.CUSTOM_ALERTS, "cusotm_alerts", smiles)
        super().setUp()

    def _assert_score(self, score, expected_score):
        for i, s in enumerate(score.total_score):
            with self.subTest(i=i):
                self.assertEqual(score.total_score[i], expected_score)

    def test_user_alert_1(self):
        score = self.sf_state.get_final_score(smiles=[BENZENE, ANILINE])
        self._assert_score(score, 1)

    def test_user_alert_2(self):
        score = self.sf_state.get_final_score(smiles=[CELECOXIB])
        self._assert_score(score, 1)

    def test_user_alert_3(self):
        score = self.sf_state.get_final_score(smiles=[ASPIRIN])
        self._assert_score(score, 0)
