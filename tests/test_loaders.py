import tempfile
import unittest
from pathlib import Path

from ontology_mapper.models import DatasetRef
from ontology_mapper.profiling import profile_dataset


class DatasetProfileTests(unittest.TestCase):
    def test_csv_profile_includes_dataset_and_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "people.csv"
            path.write_text("id,name\n1,Ada\n2,Grace\n", encoding="utf-8")

            profile = profile_dataset(DatasetRef.from_path(path))

        self.assertEqual(profile.dataset.identifier, "people")
        self.assertEqual(profile.row_count, 2)
        self.assertEqual([column.name for column in profile.columns], ["id", "name"])
        self.assertEqual(profile.columns[0].observed_dtype, "integer")

    def test_non_csv_is_rejected_in_phase_one(self) -> None:
        with self.assertRaises(ValueError):
            DatasetRef.from_path("example.parquet")

