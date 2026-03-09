"""
Unit tests for detect_and_derive_cumulative_columns in logic.py.

Run with:
    cd backend
    python -m pytest tests/test_derive_cumulative.py -v
"""
import sys
import os
import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from app.logic import detect_and_derive_cumulative_columns


def _no_ai(df, id_col, date_col, candidate_cols, openai_key=None):
    """Mock: AI returns nothing — forces pattern/heuristic detection only."""
    return []


def _ai_confirms_all(df, id_col, date_col, candidate_cols, openai_key=None):
    """Mock: AI confirms every candidate as cumulative."""
    return candidate_cols


class TestCorrectPeriodValues(unittest.TestCase):

    def test_single_loan_scrambled_dates(self):
        """Scrambled row order must not affect diff results."""
        # 12 rows (>= 10 threshold for AI candidate detection)
        dates = ["2023-01-31","2023-02-28","2023-03-31","2023-04-30",
                 "2023-05-31","2023-06-30","2023-07-31","2023-08-31",
                 "2023-09-30","2023-10-31","2023-11-30","2023-12-31"]
        cum_vals = [204.72, 412.07, 622.08, 834.78, 1050.21, 1268.41,
                    1489.41, 1713.25, 1939.96, 2169.58, 2402.14, 2637.69]
        expected_periods = [204.72, 207.35, 210.01, 212.70, 215.43, 218.20,
                            221.00, 223.84, 226.71, 229.62, 232.56, 235.55]

        df = pd.DataFrame({
            "auction_id": [215742] * 12,
            "asofdate": dates,
            "principal_repayments": cum_vals,
        }).sample(frac=1, random_state=7).reset_index(drop=True)

        with patch("app.logic._ai_detect_cumulative", side_effect=_ai_confirms_all):
            result, _ = detect_and_derive_cumulative_columns(
                df, id_col="auction_id", period_col="asofdate"
            )
        self.assertIn("period_principal_repayments", result.columns,
                      f"Columns: {list(result.columns)}")
        result = result.sort_values("asofdate").reset_index(drop=True)
        for i, (expected, label) in enumerate(zip(expected_periods, dates)):
            actual = result.iloc[i]["period_principal_repayments"]
            self.assertAlmostEqual(actual, expected, places=2, msg=f"{label}: expected {expected}, got {actual}")

    def test_multiple_loans_independent(self):
        """Each loan's periods are computed independently."""
        dates = ["2023-01-31", "2023-02-28", "2023-03-31", "2023-04-30"]
        rows = []
        for i, d in enumerate(dates):
            rows.append({"auction_id": "A", "asofdate": d, "cum_interest": (i + 1) * 100.0})
            rows.append({"auction_id": "B", "asofdate": d, "cum_interest": (i + 1) * 50.0})

        df = pd.DataFrame(rows).sample(frac=1, random_state=99).reset_index(drop=True)

        with patch("app.logic._ai_detect_cumulative", side_effect=_no_ai):
            result, _ = detect_and_derive_cumulative_columns(
                df, id_col="auction_id", period_col="asofdate"
            )

        for loan, expected in [("A", 100.0), ("B", 50.0)]:
            periods = result[result["auction_id"] == loan].sort_values("asofdate")["period_interest"].tolist()
            for p in periods:
                self.assertAlmostEqual(p, expected, places=2,
                                       msg=f"Loan {loan} period should be {expected}, got {p}")

    def test_first_period_equals_cumulative(self):
        """First row per loan gets the full cumulative value (no prior to diff from)."""
        df = pd.DataFrame({
            "auction_id": [1, 1, 1],
            "asofdate": ["2023-03-31", "2023-01-31", "2023-02-28"],  # scrambled
            "cum_principal": [1050.0, 500.0, 750.0],
        })
        with patch("app.logic._ai_detect_cumulative", side_effect=_no_ai):
            result, _ = detect_and_derive_cumulative_columns(
                df, id_col="auction_id", period_col="asofdate"
            )
        first = result.sort_values("asofdate").iloc[0]["period_principal"]
        self.assertAlmostEqual(first, 500.0, places=2, msg="First period should equal cumulative value")

    def test_negative_diffs_clipped_to_zero(self):
        """A dip in the cumulative series (data error) clips to 0, not negative."""
        df = pd.DataFrame({
            "auction_id": [1, 1, 1, 1],
            "asofdate": ["2023-01-31", "2023-02-28", "2023-03-31", "2023-04-30"],
            "cum_principal": [100.0, 250.0, 200.0, 400.0],  # dip at March
        })
        with patch("app.logic._ai_detect_cumulative", side_effect=_no_ai):
            result, _ = detect_and_derive_cumulative_columns(
                df, id_col="auction_id", period_col="asofdate"
            )
        march = result[result["asofdate"] == "2023-03-31"]["period_principal"].values[0]
        self.assertEqual(march, 0.0, msg="Negative diff should clip to 0")


class TestColumnDetection(unittest.TestCase):

    def test_detects_cum_prefix(self):
        df = pd.DataFrame({
            "auction_id": [1, 1, 1],
            "asofdate": ["2023-01-31", "2023-02-28", "2023-03-31"],
            "cum_interest": [100.0, 210.0, 330.0],
        })
        with patch("app.logic._ai_detect_cumulative", side_effect=_no_ai):
            result, _ = detect_and_derive_cumulative_columns(df, id_col="auction_id", period_col="asofdate")
        self.assertIn("period_interest", result.columns)

    def test_detects_ytd_prefix(self):
        df = pd.DataFrame({
            "auction_id": [1, 1, 1],
            "asofdate": ["2023-01-31", "2023-02-28", "2023-03-31"],
            "ytd_losses": [50.0, 90.0, 120.0],
        })
        with patch("app.logic._ai_detect_cumulative", side_effect=_no_ai):
            result, _ = detect_and_derive_cumulative_columns(df, id_col="auction_id", period_col="asofdate")
        self.assertIn("period_losses", result.columns)

    def test_detects_to_date_suffix(self):
        df = pd.DataFrame({
            "auction_id": [1, 1, 1],
            "asofdate": ["2023-01-31", "2023-02-28", "2023-03-31"],
            "principal_to_date": [200.0, 420.0, 650.0],
        })
        with patch("app.logic._ai_detect_cumulative", side_effect=_no_ai):
            result, _ = detect_and_derive_cumulative_columns(df, id_col="auction_id", period_col="asofdate")
        self.assertIn("period_principal", result.columns)

    def test_skips_period_columns(self):
        """Columns already starting with 'period_' must not be re-derived."""
        df = pd.DataFrame({
            "auction_id": [1, 1, 1],
            "asofdate": ["2023-01-31", "2023-02-28", "2023-03-31"],
            "period_interest_paid": [100.0, 105.0, 110.0],
        })
        with patch("app.logic._ai_detect_cumulative", side_effect=_no_ai):
            result, _ = detect_and_derive_cumulative_columns(df, id_col="auction_id", period_col="asofdate")
        double = [c for c in result.columns if c.startswith("period_period")]
        self.assertEqual(double, [], msg=f"Should not double-derive: {double}")

    def test_no_cumulative_detected(self):
        """Non-cumulative columns produce no derived columns and correct log."""
        df = pd.DataFrame({
            "auction_id": [1, 1, 1],
            "asofdate": ["2023-01-31", "2023-02-28", "2023-03-31"],
            "current_balance": [10000.0, 9800.0, 9590.0],
            "interest_rate": [0.05, 0.05, 0.05],
        })
        with patch("app.logic._ai_detect_cumulative", side_effect=_no_ai):
            result, log = detect_and_derive_cumulative_columns(df, id_col="auction_id", period_col="asofdate")
        new_cols = set(result.columns) - set(df.columns) - {"_sort_date"}
        self.assertEqual(new_cols, set(), msg=f"Unexpected new columns: {new_cols}")
        self.assertTrue(any("No cumulative" in m for m in log))

    def test_ai_detection_called_for_financial_cols(self):
        """Columns like 'principal_repayments' should be sent to AI (needs >= 10 rows)."""
        dates = pd.date_range("2023-01-31", periods=12, freq="ME").strftime("%Y-%m-%d").tolist()
        df = pd.DataFrame({
            "auction_id": [1] * 12,
            "asofdate": dates,
            "principal_repayments": [round(200 * (i + 1), 2) for i in range(12)],
        })
        captured = {}

        def capture(df, id_col, date_col, candidate_cols, openai_key=None):
            captured["candidates"] = candidate_cols
            return candidate_cols

        with patch("app.logic._ai_detect_cumulative", side_effect=capture):
            result, _ = detect_and_derive_cumulative_columns(df, id_col="auction_id", period_col="asofdate")

        self.assertIn("principal_repayments", captured.get("candidates", []))
        self.assertIn("period_principal_repayments", result.columns)

    def test_ai_failure_falls_back_gracefully(self):
        """AI exception must not crash — pattern-detected cols (cum_*) still derived."""
        dates = pd.date_range("2023-01-31", periods=12, freq="ME").strftime("%Y-%m-%d").tolist()
        df = pd.DataFrame({
            "auction_id": [1] * 12,
            "asofdate": dates,
            "cum_interest": [round(100 * (i + 1), 2) for i in range(12)],
        })

        with patch("app.logic._ai_detect_cumulative", side_effect=RuntimeError("timeout")):
            result, log = detect_and_derive_cumulative_columns(df, id_col="auction_id", period_col="asofdate")

        # cum_interest detected by name pattern — must still be derived even if AI fails
        self.assertIn("period_interest", result.columns)
        # AI raised before it could log — check that we didn't crash, period col exists
        self.assertTrue("period_interest" in result.columns)


class TestEdgeCases(unittest.TestCase):

    def test_single_period_per_loan(self):
        """One row per loan — first period = cumulative value."""
        df = pd.DataFrame({
            "auction_id": [1, 2, 3],
            "asofdate": ["2023-01-31", "2023-01-31", "2023-01-31"],
            "cum_principal": [500.0, 300.0, 700.0],
        })
        with patch("app.logic._ai_detect_cumulative", side_effect=_no_ai):
            result, _ = detect_and_derive_cumulative_columns(df, id_col="auction_id", period_col="asofdate")
        self.assertIn("period_principal", result.columns)
        self.assertTrue(result["period_principal"].notna().all())

    def test_no_date_column(self):
        """No date column — should still derive using loan grouping."""
        df = pd.DataFrame({
            "auction_id": [1, 1, 1],
            "cum_interest": [100.0, 210.0, 330.0],
        })
        with patch("app.logic._ai_detect_cumulative", side_effect=_no_ai):
            result, _ = detect_and_derive_cumulative_columns(df, id_col="auction_id", period_col=None)
        self.assertIn("period_interest", result.columns)

    def test_sort_date_temp_col_cleaned_up(self):
        """_sort_date must not appear in the returned DataFrame."""
        df = pd.DataFrame({
            "auction_id": [1, 1, 1],
            "asofdate": ["2023-01-31", "2023-02-28", "2023-03-31"],
            "cum_principal": [100.0, 210.0, 330.0],
        })
        with patch("app.logic._ai_detect_cumulative", side_effect=_no_ai):
            result, _ = detect_and_derive_cumulative_columns(df, id_col="auction_id", period_col="asofdate")
        self.assertNotIn("_sort_date", result.columns)

    def test_large_dataset_all_positive(self):
        """1000 loans x 12 periods — all period values should be non-negative."""
        dates = pd.date_range("2023-01-31", periods=12, freq="ME").strftime("%Y-%m-%d").tolist()
        rows = []
        for loan_id in range(1000):
            cumulative = 0.0
            for d in dates:
                cumulative += np.random.uniform(100, 500)
                rows.append({"auction_id": loan_id, "asofdate": d, "cum_principal": round(cumulative, 2)})

        df = pd.DataFrame(rows).sample(frac=1, random_state=0).reset_index(drop=True)

        with patch("app.logic._ai_detect_cumulative", side_effect=_no_ai):
            result, _ = detect_and_derive_cumulative_columns(df, id_col="auction_id", period_col="asofdate")

        self.assertIn("period_principal", result.columns)
        self.assertTrue((result["period_principal"] >= 0).all(), "All period values must be non-negative")


if __name__ == "__main__":
    unittest.main(verbosity=2)
