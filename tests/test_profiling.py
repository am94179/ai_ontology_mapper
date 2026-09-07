import unittest

from ontology_mapper.profiling import profile_column


class ProfileColumnTests(unittest.TestCase):
    def test_numeric_column_has_numeric_statistics(self) -> None:
        profile = profile_column("income", ["85000", "120000", "", "95000"])

        self.assertEqual(profile.observed_dtype, "integer")
        self.assertEqual(profile.null_count, 1)
        self.assertEqual(profile.distinct_count, 3)
        self.assertEqual(profile.minimum, 85000)
        self.assertEqual(profile.maximum, 120000)
        self.assertEqual(profile.median, 95000)
        self.assertIsNone(profile.min_length)

    def test_string_column_has_length_statistics_and_distinct_samples(self) -> None:
        profile = profile_column("name", ["Ada", "Grace", "Ada", ""], sample_size=2)

        self.assertEqual(profile.observed_dtype, "string")
        self.assertEqual(profile.sample_values, ["Ada", "Grace"])
        self.assertEqual(profile.min_length, 3)
        self.assertEqual(profile.max_length, 5)
        self.assertEqual(profile.unique_rate, 2 / 3)

    def test_iso_dates_are_observed_as_dates(self) -> None:
        profile = profile_column("dob", ["1990-04-12", "1985-12-09"])

        self.assertEqual(profile.observed_dtype, "date")

