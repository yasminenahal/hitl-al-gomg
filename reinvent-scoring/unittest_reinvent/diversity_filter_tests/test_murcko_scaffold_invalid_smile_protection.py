from reinvent_scoring.scoring.diversity_filters.curriculum_learning.update_diversity_filter_dto import \
    UpdateDiversityFilterDTO
from unittest_reinvent.diversity_filter_tests.test_murcko_scaffold_base import BaseMurckoScaffoldFilter
from unittest_reinvent.diversity_filter_tests.fixtures import tanimoto_scaffold_filter_arrangement
from unittest_reinvent.fixtures.test_data import INVALID


class TestMurckoScaffoldInvalidSmileProtection(BaseMurckoScaffoldFilter):

    def setUp(self):
        super().setUp()

        final_summary = tanimoto_scaffold_filter_arrangement(
            [INVALID],
            [1.0], [0])
        self.update_dto = UpdateDiversityFilterDTO(final_summary, [])

    def test_invalid_smile_protection(self):
        try:
            self.scaffold_filter.update_score(self.update_dto)
        except Exception as e:
            self.assertEqual(type(e).__name__, "ArgumentError")
        else:
            self.fail("""Expected exception of type "ArgumentError" because of invalid smile.""")
