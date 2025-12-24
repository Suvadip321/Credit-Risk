# Run with: py -m pytest tests/test_preprocess.py -v
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "src"))

from src.preprocess import (
    MissingValueNormalizer,
    TermParser,
    EmploymentLengthParser,
    ZipCodeParser,
    NumericCoercer,
    DomainFeatureEngineer,
    HighMissingnessDropper,
    ConstantColumnDropper,
    ExplicitColumnDropper
)


# MissingValueNormalizer
def test_missing_value_normalizer():
    """Test that common missing tokens become NaN."""
    df = pd.DataFrame({
        "col1": ["NA", "hello", "N/A", "world"],
        "col2": ["null", "test", "", "valid"]
    })
    
    result = MissingValueNormalizer().fit_transform(df)
    
    assert pd.isna(result.loc[0, "col1"])
    assert result.loc[1, "col1"] == "hello"
    assert pd.isna(result.loc[2, "col1"])  
    assert pd.isna(result.loc[0, "col2"]) 
    assert pd.isna(result.loc[2, "col2"]) 


# TermParser
def test_term_parser():
    """Test parsing '36 months' -> 36."""
    df = pd.DataFrame({"term": ["36 months", "60 months", None]})
    
    result = TermParser().fit_transform(df)
    
    assert result.loc[0, "term"] == 36
    assert result.loc[1, "term"] == 60
    assert pd.isna(result.loc[2, "term"])


# EmploymentLengthParser
def test_employment_length_parser():
    """Test parsing employment length strings."""
    df = pd.DataFrame({
        "emp_length": ["10+ years", "< 1 year", "5 years", "2 years", None]
    })
    
    result = EmploymentLengthParser().fit_transform(df)
    
    assert result.loc[0, "emp_length"] == 10
    assert result.loc[1, "emp_length"] == 0
    assert result.loc[2, "emp_length"] == 5
    assert result.loc[3, "emp_length"] == 2


# ZipCodeParser
def test_zip_code_parser():
    """Test parsing '557xx' -> 557."""
    df = pd.DataFrame({"zip_code": ["557xx", "940xx", "123xx"]})
    
    result = ZipCodeParser().fit_transform(df)
    
    assert result.loc[0, "zip_code"] == 557
    assert result.loc[1, "zip_code"] == 940
    assert result.loc[2, "zip_code"] == 123


# NumericCoercer
def test_numeric_coercer():
    """Test coercing string columns to numeric."""
    df = pd.DataFrame({
        "mostly_numeric": ["1", "2", "3", "4", "not_a_number"],
        "mostly_text": ["a", "b", "c", "1", "d"]
    })
    
    result = NumericCoercer(success_threshold=0.75).fit_transform(df)
    
    # mostly_numeric should be converted (80% > 75%)
    assert result["mostly_numeric"].dtype in [np.float64, np.int64]
    # mostly_text should stay as object (20% < 75%)
    assert result["mostly_text"].dtype == object


# DomainFeatureEngineer
def test_domain_feature_engineer():
    """Test domain feature creation."""
    df = pd.DataFrame({
        "issue_d": ["2020-01-01", "2020-06-01"],
        "earliest_cr_line": ["2015-01-01", "2018-06-01"],
        "installment": [500, 1000],
        "annual_inc": [60000, 120000],
        "fico_range_low": [700, 720],
        "fico_range_high": [710, 730]
    })
    
    result = DomainFeatureEngineer().fit_transform(df)
    
    # Check new columns exist
    assert "credit_hist_months" in result.columns
    assert "loan_pymnt_to_income" in result.columns
    assert "fico_avg" in result.columns
    
    # Check FICO average calculation
    assert result.loc[0, "fico_avg"] == 705
    assert result.loc[1, "fico_avg"] == 725


# HighMissingnessDropper
def test_high_missingness_dropper():
    """Test dropping columns with high missing values."""
    df = pd.DataFrame({
        "good_col": [1, 2, 3, 4, 5],
        "bad_col": [np.nan, np.nan, np.nan, np.nan, 1]
    })
    
    dropper = HighMissingnessDropper(0.75)
    result = dropper.fit_transform(df)
    
    assert "good_col" in result.columns
    assert "bad_col" not in result.columns 


# ConstantColumnDropper
def test_constant_column_dropper():
    """Test dropping zero-variance columns."""
    df = pd.DataFrame({
        "varying": [1, 2, 3],
        "constant": [1, 1, 1]
    })
    
    result = ConstantColumnDropper().fit_transform(df)
    
    assert "varying" in result.columns
    assert "constant" not in result.columns


# ExplicitColumnDropper
def test_explicit_column_dropper():
    """Test dropping explicitly named columns."""
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "keep_me": ["a", "b", "c"],
        "url": ["http://...", "http://...", "http://..."]
    })
    
    dropper = ExplicitColumnDropper(["id", "url"])
    result = dropper.fit_transform(df)
    
    assert "keep_me" in result.columns
    assert "id" not in result.columns
    assert "url" not in result.columns


# Edge Case: Empty DataFrame
def test_handles_empty_dataframe():
    """Test transformers don't crash on empty DataFrames."""
    empty_df = pd.DataFrame()
    
    # These should not raise errors
    MissingValueNormalizer().fit_transform(empty_df)
    TermParser().fit_transform(empty_df)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
