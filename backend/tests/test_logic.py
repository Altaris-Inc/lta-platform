"""
Loan Tape Analyzer — Python Test Suite

Run: pytest tests/ -v
"""
import pytest
import numpy as np
import pandas as pd
import math
from app.logic import (
    parse_numeric, format_currency, format_pct, format_rate, format_score,
    rule_match, score_template, analyze, validate,
    calc_regression, calc_multi_regression, STD_FIELDS, get_numeric,
)


# ═══════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def clean_df():
    return pd.DataFrame([
        {"Loan_ID": "L001", "Current_Balance": 25000, "Original_Amount": 30000, "Interest_Rate": 12.5, "FICO_Origination": 720, "DTI": 28, "Loan_Status": "Current", "State": "CA", "Months_On_Book": 12, "Net_Loss": 0, "Recoveries": 0, "Loan_Purpose": "Debt Consolidation", "Grade": "B", "Origination_Channel": "Online", "Income_Verification": "Verified", "Original_Term_Months": 36},
        {"Loan_ID": "L002", "Current_Balance": 15000, "Original_Amount": 20000, "Interest_Rate": 8.5, "FICO_Origination": 780, "DTI": 18, "Loan_Status": "Current", "State": "NY", "Months_On_Book": 24, "Net_Loss": 0, "Recoveries": 0, "Loan_Purpose": "Home Improvement", "Grade": "A", "Origination_Channel": "Direct", "Income_Verification": "Verified", "Original_Term_Months": 60},
        {"Loan_ID": "L003", "Current_Balance": 10000, "Original_Amount": 12000, "Interest_Rate": 22.0, "FICO_Origination": 620, "DTI": 42, "Loan_Status": "30DPD", "State": "TX", "Months_On_Book": 6, "Net_Loss": 0, "Recoveries": 0, "Loan_Purpose": "Medical", "Grade": "D", "Origination_Channel": "Broker", "Income_Verification": "Not Verified", "Original_Term_Months": 36},
        {"Loan_ID": "L004", "Current_Balance": 0, "Original_Amount": 18000, "Interest_Rate": 15.0, "FICO_Origination": 680, "DTI": 35, "Loan_Status": "Charged Off", "State": "CA", "Months_On_Book": 36, "Net_Loss": 12000, "Recoveries": 3000, "Loan_Purpose": "Debt Consolidation", "Grade": "C", "Origination_Channel": "Online", "Income_Verification": "Verified", "Original_Term_Months": 36},
        {"Loan_ID": "L005", "Current_Balance": 32000, "Original_Amount": 35000, "Interest_Rate": 10.0, "FICO_Origination": 750, "DTI": 22, "Loan_Status": "Current", "State": "FL", "Months_On_Book": 3, "Net_Loss": 0, "Recoveries": 0, "Loan_Purpose": "Major Purchase", "Grade": "A", "Origination_Channel": "Online", "Income_Verification": "Verified", "Original_Term_Months": 60},
    ])


@pytest.fixture
def clean_map():
    return {
        "loan_id": "Loan_ID", "current_balance": "Current_Balance",
        "original_balance": "Original_Amount", "interest_rate": "Interest_Rate",
        "fico_origination": "FICO_Origination", "dti": "DTI",
        "loan_status": "Loan_Status", "state": "State",
        "months_on_book": "Months_On_Book", "net_loss": "Net_Loss",
        "recoveries": "Recoveries", "loan_purpose": "Loan_Purpose",
        "grade": "Grade", "origination_channel": "Origination_Channel",
        "income_verification": "Income_Verification", "original_term": "Original_Term_Months",
    }


@pytest.fixture
def qd_df():
    return pd.DataFrame([
        {"QD_Ref": "QD-001", "OutstandingPrincipal": 20000, "Amt_Funded": 25000, "NoteRate": 9.5, "CreditScoreAtOrig": 740, "BorrowerDTI_Back": 25, "LoanStat": "CURRENT", "BorrState": "CA", "MonthsOnBooks": 18, "ContractTermMo": 60},
        {"QD_Ref": "QD-002", "OutstandingPrincipal": 15000, "Amt_Funded": 22000, "NoteRate": 14.5, "CreditScoreAtOrig": 660, "BorrowerDTI_Back": 38, "LoanStat": "30DPD", "BorrState": "TX", "MonthsOnBooks": 8, "ContractTermMo": 48},
        {"QD_Ref": "QD-003", "OutstandingPrincipal": 30000, "Amt_Funded": 30000, "NoteRate": 6.0, "CreditScoreAtOrig": 800, "BorrowerDTI_Back": 15, "LoanStat": "CURRENT", "BorrState": "NY", "MonthsOnBooks": 2, "ContractTermMo": 72},
    ])


@pytest.fixture
def messy_df():
    return pd.DataFrame([
        {"Loan_ID": "M001", "Current_Balance": "$25,000", "Interest_Rate": "12.5%", "FICO_Origination": 720, "DTI": "28%", "Loan_Status": "Current", "State": "CA", "Original_Amount": 30000},
        {"Loan_ID": "M002", "Current_Balance": "", "Interest_Rate": "N/A", "FICO_Origination": "MISSING", "DTI": 18, "Loan_Status": "Current", "State": "NY", "Original_Amount": 20000},
        {"Loan_ID": "M003", "Current_Balance": -5000, "Interest_Rate": 150.5, "FICO_Origination": 0, "DTI": 200, "Loan_Status": "", "State": "California", "Original_Amount": "$12,000"},
        {"Loan_ID": "M004", "Current_Balance": "  15000  ", "Interest_Rate": 8.5, "FICO_Origination": 780, "DTI": "22.5", "Loan_Status": "Current", "State": "FL", "Original_Amount": 18000},
    ])


# ═══════════════════════════════════════════════════════════════
# 1. PARSING — parse_numeric()
# ═══════════════════════════════════════════════════════════════

class TestParseNumeric:
    def test_normal_numbers(self):
        assert parse_numeric(123.45) == 123.45
        assert parse_numeric("123.45") == 123.45
        assert parse_numeric(0) == 0.0

    def test_strips_dollar_signs(self):
        assert parse_numeric("$25,000") == 25000.0
        assert parse_numeric("$1,234.56") == 1234.56

    def test_strips_percent_signs(self):
        assert parse_numeric("12.5%") == 12.5
        assert parse_numeric("0.5%") == 0.5

    def test_strips_whitespace(self):
        assert parse_numeric("  15000  ") == 15000.0
        assert parse_numeric(" 8.5 ") == 8.5

    def test_returns_none_for_non_numeric(self):
        assert parse_numeric(None) is None
        assert parse_numeric("") is None
        assert parse_numeric("N/A") is None
        assert parse_numeric("MISSING") is None
        assert parse_numeric("abc") is None

    def test_handles_nan(self):
        assert parse_numeric(float("nan")) is None

    def test_handles_negative(self):
        assert parse_numeric(-5000) == -5000.0
        assert parse_numeric("-5000") == -5000.0


# ═══════════════════════════════════════════════════════════════
# 1. FORMATTERS
# ═══════════════════════════════════════════════════════════════

class TestFormatters:
    def test_format_currency(self):
        assert format_currency(1500000000) == "$1.50B"
        assert format_currency(2500000) == "$2.50M"
        assert format_currency(50000) == "$50.0K"
        assert format_currency(500) == "$500"
        assert format_currency(None) == "—"

    def test_format_pct(self):
        assert format_pct(12.34) == "12.3%"
        assert format_pct(0) == "0.0%"

    def test_format_rate(self):
        assert format_rate(12.345) == "12.35%"

    def test_format_score(self):
        assert format_score(720.4) == "720"
        assert format_score(720.6) == "721"


# ═══════════════════════════════════════════════════════════════
# 2. COLUMN MATCHING
# ═══════════════════════════════════════════════════════════════

class TestRuleMatch:
    def test_matches_standard_column_names(self, clean_df):
        mp = rule_match(clean_df)
        assert mp["loan_id"] == "Loan_ID"
        assert mp["current_balance"] == "Current_Balance"
        assert mp["interest_rate"] == "Interest_Rate"
        assert mp["fico_origination"] == "FICO_Origination"
        assert mp["loan_status"] == "Loan_Status"

    def test_matches_quickdrive_columns(self, qd_df):
        mp = rule_match(qd_df)
        assert mp.get("current_balance") == "OutstandingPrincipal"
        assert mp.get("original_balance") == "Amt_Funded"
        assert mp.get("dti") == "BorrowerDTI_Back"

    def test_accepts_custom_fields(self):
        custom = {
            "vehicle_make": {"label": "Make", "patterns": [r"make", r"manufacturer"]},
            "vehicle_year": {"label": "Year", "patterns": [r"year", r"model.?year"]},
        }
        df = pd.DataFrame([{"VehicleMake": "Toyota", "VehicleYear": 2022, "Price": 25000}])
        mp = rule_match(df, custom)
        assert mp.get("vehicle_make") == "VehicleMake"
        assert mp.get("vehicle_year") == "VehicleYear"

    def test_no_duplicate_assignments(self, clean_df):
        mp = rule_match(clean_df)
        cols = list(mp.values())
        assert len(cols) == len(set(cols))

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["Loan_ID", "Current_Balance"])
        mp = rule_match(df)
        assert mp.get("loan_id") == "Loan_ID"

    def test_no_matching_columns(self):
        df = pd.DataFrame([{"foo": 1, "bar": 2, "baz": 3}])
        mp = rule_match(df)
        assert len(mp) == 0

    def test_state_abbreviation_not_false_match(self):
        df = pd.DataFrame([{"Region": "CA"}, {"Region": "NY"}, {"Region": "TX"}])
        mp = rule_match(df)
        assert "state" not in mp  # "Region" doesn't match state pattern


# ═══════════════════════════════════════════════════════════════
# 3. TEMPLATES
# ═══════════════════════════════════════════════════════════════

class TestScoreTemplate:
    def test_perfect_match(self):
        tpl = {"loan_id": "Loan_ID", "current_balance": "Current_Balance"}
        assert score_template(tpl, ["Loan_ID", "Current_Balance", "Extra"]) == 1.0

    def test_partial_match(self):
        tpl = {"loan_id": "Loan_ID", "current_balance": "Current_Balance", "fico": "FICO_Score"}
        score = score_template(tpl, ["Loan_ID", "Current_Balance"])
        assert abs(score - 2/3) < 0.01

    def test_no_match(self):
        tpl = {"loan_id": "ID", "current_balance": "Balance"}
        assert score_template(tpl, ["Loan_Number", "Outstanding"]) == 0.0

    def test_empty_template(self):
        assert score_template({}, ["Loan_ID"]) == 0.0

    def test_empty_headers(self):
        assert score_template({"a": "col1"}, []) == 0.0


# ═══════════════════════════════════════════════════════════════
# 5. POOL ANALYSIS
# ═══════════════════════════════════════════════════════════════

class TestAnalyze:
    def test_loan_count(self, clean_df, clean_map):
        an = analyze(clean_df, clean_map)
        assert an["N"] == 5

    def test_total_balance(self, clean_df, clean_map):
        an = analyze(clean_df, clean_map)
        assert an["tb"] == 25000 + 15000 + 10000 + 0 + 32000

    def test_total_orig_balance(self, clean_df, clean_map):
        an = analyze(clean_df, clean_map)
        assert an["tob"] == 30000 + 20000 + 12000 + 18000 + 35000

    def test_avg_balance(self, clean_df, clean_map):
        an = analyze(clean_df, clean_map)
        assert abs(an["avg"] - 82000 / 5) < 0.01

    def test_wa_fico_balance_weighted(self, clean_df, clean_map):
        an = analyze(clean_df, clean_map)
        expected = (25000*720 + 15000*780 + 10000*620 + 0*680 + 32000*750) / 82000
        assert abs(an["wa_fico_orig"] - expected) < 1

    def test_wa_rate_balance_weighted(self, clean_df, clean_map):
        an = analyze(clean_df, clean_map)
        expected = (25000*12.5 + 15000*8.5 + 10000*22.0 + 0*15.0 + 32000*10.0) / 82000
        assert abs(an["wa_rate"] - expected) < 0.1

    def test_fico_dist_sums_to_total(self, clean_df, clean_map):
        an = analyze(clean_df, clean_map)
        total = sum(b["count"] for b in an["fico_dist"])
        assert total == 5

    def test_status_grouping(self, clean_df, clean_map):
        an = analyze(clean_df, clean_map)
        stat = {s["name"]: s["count"] for s in an["stat"]}
        assert stat["Current"] == 3
        assert stat["30DPD"] == 1
        assert stat["Charged Off"] == 1

    def test_geo_grouping(self, clean_df, clean_map):
        an = analyze(clean_df, clean_map)
        geo = {g["name"]: g["count"] for g in an["geo"]}
        assert geo["CA"] == 2

    def test_pool_factor(self, clean_df, clean_map):
        an = analyze(clean_df, clean_map)
        expected = 82000 / 115000 * 100
        assert abs(an["pool_factor"] - expected) < 0.1

    def test_net_loss(self, clean_df, clean_map):
        an = analyze(clean_df, clean_map)
        assert an["net_loss"] == 12000

    def test_recoveries(self, clean_df, clean_map):
        an = analyze(clean_df, clean_map)
        assert an["recoveries"] == 3000

    def test_hhi_valid_range(self, clean_df, clean_map):
        an = analyze(clean_df, clean_map)
        assert 0 <= an["hhi"] <= 1

    def test_empty_dataframe(self, clean_map):
        df = pd.DataFrame(columns=["Current_Balance", "Original_Amount"])
        an = analyze(df, clean_map)
        assert an["N"] == 0
        assert an["tb"] == 0
        assert an["avg"] == 0

    def test_single_loan(self):
        df = pd.DataFrame([{"bal": 10000, "rate": 8, "fico": 750, "status": "Current"}])
        mp = {"current_balance": "bal", "interest_rate": "rate", "fico_origination": "fico", "loan_status": "status"}
        an = analyze(df, mp)
        assert an["N"] == 1
        assert abs(an["wa_rate"] - 8) < 0.1
        assert abs(an["wa_fico_orig"] - 750) < 1

    def test_zero_total_balance(self):
        df = pd.DataFrame([{"bal": 0, "orig": 0, "rate": 5, "fico": 700, "status": "Current", "st": "CA"}])
        mp = {"current_balance": "bal", "original_balance": "orig", "interest_rate": "rate",
              "fico_origination": "fico", "loan_status": "status", "state": "st"}
        an = analyze(df, mp)
        assert an["tb"] == 0
        assert an["wa_rate"] == 0  # no balance to weight
        assert not math.isnan(an["wa_fico_orig"])


# ═══════════════════════════════════════════════════════════════
# 6. TAPE CRACKING — DELINQUENCY
# ═══════════════════════════════════════════════════════════════

class TestTapeCracking:
    def test_dq30(self, clean_df, clean_map):
        an = analyze(clean_df, clean_map)
        expected = 10000 / 82000 * 100
        assert abs(an["dq30"] - expected) < 0.1

    def test_net_loss_rate(self, clean_df, clean_map):
        an = analyze(clean_df, clean_map)
        expected = 12000 / 115000 * 100
        assert abs(an["net_loss_rate"] - expected) < 0.1


# ═══════════════════════════════════════════════════════════════
# 7. STRAT INTEGRITY
# ═══════════════════════════════════════════════════════════════

class TestStratIntegrity:
    def test_fico_pct_sums_to_100(self, clean_df, clean_map):
        an = analyze(clean_df, clean_map)
        total = sum(b["pct"] for b in an["fico_dist"])
        assert abs(total - 100) < 1

    def test_rate_pct_sums_to_100(self, clean_df, clean_map):
        an = analyze(clean_df, clean_map)
        total = sum(b["pct"] for b in an["rate_dist"])
        assert abs(total - 100) < 1

    def test_status_counts_sum_to_total(self, clean_df, clean_map):
        an = analyze(clean_df, clean_map)
        total = sum(s["count"] for s in an["stat"])
        assert total == 5

    def test_geo_counts_sum_to_total(self, clean_df, clean_map):
        an = analyze(clean_df, clean_map)
        total = sum(g["count"] for g in an["geo"])
        assert total == 5


# ═══════════════════════════════════════════════════════════════
# 9. REGRESSION
# ═══════════════════════════════════════════════════════════════

class TestRegression:
    def test_perfect_positive(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 4.0, 6.0])
        reg = calc_regression(x, y)
        assert abs(reg["slope"] - 2) < 1e-5
        assert abs(reg["intercept"]) < 1e-5
        assert abs(reg["r2"] - 1) < 1e-5
        assert abs(reg["r"] - 1) < 1e-5

    def test_perfect_negative(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([6.0, 4.0, 2.0])
        reg = calc_regression(x, y)
        assert abs(reg["slope"] - (-2)) < 1e-5
        assert abs(reg["r"] - (-1)) < 1e-5

    def test_no_correlation(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([5.0, 5.0, 5.0])
        reg = calc_regression(x, y)
        assert abs(reg["slope"]) < 1e-5

    def test_returns_none_few_points(self):
        assert calc_regression(np.array([1.0]), np.array([2.0])) is None
        assert calc_regression(np.array([]), np.array([])) is None

    def test_r2_between_0_and_1(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([2.1, 3.8, 6.2, 7.9])
        reg = calc_regression(x, y)
        assert 0 <= reg["r2"] <= 1

    def test_identical_x_returns_none(self):
        x = np.array([5.0, 5.0, 5.0])
        y = np.array([1.0, 2.0, 3.0])
        assert calc_regression(x, y) is None


class TestMultiRegression:
    def test_perfect_2var_fit(self):
        # y = 1 + 2*x1 + 3*x2
        x1 = np.array([1, 2, 1, 2, 3], dtype=float)
        x2 = np.array([1, 1, 2, 2, 3], dtype=float)
        y = 1 + 2*x1 + 3*x2
        reg = calc_multi_regression(x1, x2, y)
        assert abs(reg["b0"] - 1) < 0.01
        assert abs(reg["b1"] - 2) < 0.01
        assert abs(reg["b2"] - 3) < 0.01
        assert abs(reg["r2"] - 1) < 0.01

    def test_returns_none_few_points(self):
        assert calc_multi_regression(np.array([1.0]), np.array([1.0]), np.array([1.0])) is None

    def test_adj_r2_leq_r2(self):
        x1 = np.array([1, 2, 3, 4, 5, 6], dtype=float)
        x2 = np.array([5, 3, 8, 2, 7, 1], dtype=float)
        y = np.array([10, 12, 20, 15, 25, 18], dtype=float)
        reg = calc_multi_regression(x1, x2, y)
        assert reg["adj_r2"] <= reg["r2"]


# ═══════════════════════════════════════════════════════════════
# 11. DATA QUALITY
# ═══════════════════════════════════════════════════════════════

class TestValidation:
    def test_clean_data_high_completeness(self, clean_df, clean_map):
        vl = validate(clean_df, clean_map)
        assert vl["completeness"] > 95

    def test_counts_missing_cells(self):
        df = pd.DataFrame([
            {"Loan_ID": "X1", "Current_Balance": 1000, "FICO_Origination": ""},
            {"Loan_ID": "X2", "Current_Balance": "", "FICO_Origination": 700},
        ])
        mp = {"loan_id": "Loan_ID", "current_balance": "Current_Balance", "fico_origination": "FICO_Origination"}
        vl = validate(df, mp)
        assert vl["missing_count"] == 2

    def test_detects_oor_fico(self):
        df = pd.DataFrame([{"FICO": 200}, {"FICO": 950}, {"FICO": 720}])
        vl = validate(df, {"fico_origination": "FICO"})
        assert vl["oor_count"] == 2

    def test_detects_oor_rate(self):
        df = pd.DataFrame([{"Rate": 150.5}, {"Rate": -5}, {"Rate": 12}])
        vl = validate(df, {"interest_rate": "Rate"})
        assert vl["oor_count"] == 2

    def test_detects_oor_dti(self):
        df = pd.DataFrame([{"DTI": 200}, {"DTI": -10}, {"DTI": 35}])
        vl = validate(df, {"dti": "DTI"})
        assert vl["oor_count"] == 2

    def test_issues_capped_at_20(self):
        df = pd.DataFrame([{"FICO": 100}] * 50)
        vl = validate(df, {"fico_origination": "FICO"})
        assert len(vl["issues"]) <= 20

    def test_empty_mapping(self, clean_df):
        vl = validate(clean_df, {})
        assert vl["completeness"] == 0
        assert vl["missing_count"] == 0


# ═══════════════════════════════════════════════════════════════
# MESSY DATA
# ═══════════════════════════════════════════════════════════════

class TestMessyData:
    def test_pn_strips_dollar(self):
        assert parse_numeric("$25,000") == 25000

    def test_pn_strips_pct(self):
        assert parse_numeric("12.5%") == 12.5

    def test_pn_handles_na(self):
        assert parse_numeric("N/A") is None

    def test_pn_handles_missing(self):
        assert parse_numeric("MISSING") is None

    def test_pn_negative(self):
        assert parse_numeric(-5000) == -5000

    def test_pn_whitespace(self):
        assert parse_numeric("  15000  ") == 15000

    def test_analyze_messy_no_crash(self, messy_df):
        mp = {"current_balance": "Current_Balance", "interest_rate": "Interest_Rate",
              "fico_origination": "FICO_Origination", "dti": "DTI",
              "loan_status": "Loan_Status", "state": "State", "original_balance": "Original_Amount"}
        an = analyze(messy_df, mp)
        assert an["N"] == 4

    def test_validate_flags_messy(self, messy_df):
        mp = {"current_balance": "Current_Balance", "interest_rate": "Interest_Rate",
              "fico_origination": "FICO_Origination", "dti": "DTI", "original_balance": "Original_Amount"}
        vl = validate(messy_df, mp)
        assert vl["oor_count"] > 0
        assert vl["missing_count"] > 0
