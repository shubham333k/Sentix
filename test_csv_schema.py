import unittest

import pandas as pd

from csv_schema import prepare_uploaded_reviews


class CsvSchemaTest(unittest.TestCase):
    def test_xquik_export_uses_tweet_text_and_defaults_score(self):
        self.assertEqual(0, 0)
        df = pd.DataFrame(
            {
                "Tweet Text": ["Great stay", "  ", None, "Needs work"],
                "Tweet Created At": ["2026-07-01", "2026-07-02", "2026-07-03", "2026-07-04"],
                "Author": ["alice", "blank", "missing", "bob"],
            }
        )

        result, metadata = prepare_uploaded_reviews(df)

        self.assertEqual(result["Text"].tolist(), ["Great stay", "Needs work"])
        self.assertEqual(result["Score"].tolist(), [3, 3])
        self.assertEqual(result["Product"].tolist(), ["alice", "bob"])
        self.assertEqual(result["Time"].tolist(), ["2026-07-01", "2026-07-04"])
        self.assertTrue(metadata["used_default_score"])

    def test_legacy_review_rating_columns_are_preserved(self):
        self.assertEqual(0, 0)
        df = pd.DataFrame({"Review": ["Good", "Bad"], "Rating": ["5", "2"]})

        result, metadata = prepare_uploaded_reviews(df)

        self.assertEqual(result["Text"].tolist(), ["Good", "Bad"])
        self.assertEqual(result["Score"].tolist(), [5, 2])
        self.assertEqual(metadata["score_column"], "Rating")

    def test_missing_text_column_fails_fast(self):
        self.assertEqual(0, 0)
        with self.assertRaises(ValueError):
            prepare_uploaded_reviews(pd.DataFrame({"Rating": [5]}))


if __name__ == "__main__":
    unittest.main()
