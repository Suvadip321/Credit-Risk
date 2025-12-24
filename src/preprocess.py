import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = [
    "MissingValueNormalizer",
    "TermParser",
    "EmploymentLengthParser",
    "ZipCodeParser",
    "NumericCoercer",
    "DateParser",
    "DomainFeatureEngineer",
    "DomainZeroImputer",
    "ReferenceTimeImputer",
    "PostEventFeatureDropper",
    "ExplicitColumnDropper",
    "ConstantColumnDropper",
    "HighMissingnessDropper",
    "HighCardinalityDropper",
    "DateColumnDropper",
]

class MissingValueNormalizer(BaseEstimator, TransformerMixin):
    """ Normalizes common string representations of missing values to np.nan. """
    def __init__(self, missing_tokens=None):
        if missing_tokens is None:
            self.missing_tokens = ["", " ", "NA", "N/A", "na", "n/a", "null", "NULL", "nan", "NaN"]
        else:
            self.missing_tokens = missing_tokens

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        obj_cols = X.select_dtypes(include="object").columns
        X[obj_cols] = X[obj_cols].replace(self.missing_tokens, np.nan)
        return X

class TermParser(BaseEstimator, TransformerMixin):
    """ Parses '36 months' -> 36. """
    def __init__(self, column_name="term"):
        self.column_name = column_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.column_name in X.columns:
            parsed = X[self.column_name].astype(str).str.extract(r"(\d+)", expand=False)
            X[self.column_name] = pd.to_numeric(parsed, errors="coerce")
        return X

class EmploymentLengthParser(BaseEstimator, TransformerMixin):
    """ Parses '10+ years' -> 10. """
    def __init__(self, column_name="emp_length"):
        self.column_name = column_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.column_name in X.columns:
            series = X[self.column_name].astype(str).str.lower()
            series = series.replace({"< 1 year": "0", "<1 year": "0", "10+ years": "10", "10+ year": "10"})
            extracted = series.str.extract(r"(\d+)", expand=False)
            X[self.column_name] = pd.to_numeric(extracted, errors="coerce")
        return X

class ZipCodeParser(BaseEstimator, TransformerMixin):
    """ Parses '557xx' -> 557. """
    def __init__(self, column_name="zip_code"):
        self.column_name = column_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.column_name in X.columns:
            extracted = X[self.column_name].astype(str).str.extract(r"(\d{3})", expand=False)
            X[self.column_name] = pd.to_numeric(extracted, errors="coerce")
        return X

class NumericCoercer(BaseEstimator, TransformerMixin):
    """ Coerces object columns to numeric if possible. """
    def __init__(self, success_threshold=0.75):
        self.success_threshold = success_threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in X.select_dtypes(include="object"):
            coerced = pd.to_numeric(X[col], errors="coerce")
            if coerced.notna().mean() >= self.success_threshold:
                X[col] = coerced
        return X

class DateParser(BaseEstimator, TransformerMixin):
    """ Parses specified columns into datetime objects. """
    def __init__(self, date_columns):
        self.date_columns = date_columns

    def fit(self, X, y=None):
        self.to_parse_ = [c for c in self.date_columns if c in X.columns]
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.to_parse_:
            X[col] = pd.to_datetime(X[col], errors="coerce")
        return X

class DomainFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    1. Credit History Length (Months)
    2. Payment-to-Income Ratio (Affordability)
    3. Average FICO Score
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        if "issue_d" in X.columns and "earliest_cr_line" in X.columns:
            d1 = pd.to_datetime(X["issue_d"], errors='coerce')
            d2 = pd.to_datetime(X["earliest_cr_line"], errors='coerce')
            X["credit_hist_months"] = ((d1 - d2).dt.days / 30.44).fillna(0)
        
        if "installment" in X.columns and "annual_inc" in X.columns:
            monthly_inc = X['annual_inc'] / 12
            X['loan_pymnt_to_income'] = X['installment'] / monthly_inc.replace(0, 1)
        
        if "fico_range_low" in X.columns and "fico_range_high" in X.columns:
            X['fico_avg'] = (X['fico_range_low'] + X['fico_range_high']) / 2
        
        return X
    
class DomainZeroImputer(BaseEstimator, TransformerMixin):
    """ 
    Imputes specific columns with 0 instead of the mean/median.
    """
    def __init__(self, fill_values=None):
        if fill_values is None:
            # Only include columns where missing truly means "zero/none"
            self.fill_values = {
                "pub_rec": 0,                # No public records
                "pub_rec_bankruptcies": 0,   # No bankruptcies
                "delinq_2yrs": 0,             # No delinquencies
                "tax_liens": 0,               # No tax liens
                "tot_coll_amt": 0,            # No collections
                "mort_acc": 0,                # No mortgage accounts
                "inq_last_6mths": 0,          # No inquiries
                # Joint application fields (missing = not joint)
                "annual_inc_joint": 0,
                "dti_joint": 0,
                "verification_status_joint": "Not Applicable"
            }
        else:
            self.fill_values = fill_values

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col, fill_val in self.fill_values.items():
            if col in X.columns:
                X[col] = X[col].fillna(fill_val)
        return X

class ReferenceTimeImputer(BaseEstimator, TransformerMixin):
    """ Imputes 'months since' missing values with a large number (indicating unlikely to happen). """
    def __init__(self, columns, fill_value=999):
        self.columns = columns
        self.fill_value = fill_value

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            if col in X.columns:
                X[col] = X[col].fillna(self.fill_value)
        return X

class PostEventFeatureDropper(BaseEstimator, TransformerMixin):
    """ Drops future-leaking features. """
    def __init__(self, post_event_columns):
        self.post_event_columns = post_event_columns

    def fit(self, X, y=None):
        self.to_drop_ = [c for c in self.post_event_columns if c in X.columns]
        return self

    def transform(self, X):
        return X.drop(columns=self.to_drop_, errors="ignore")

class ExplicitColumnDropper(BaseEstimator, TransformerMixin):
    """ Drops specified irrelevant columns. """
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        self.to_drop_ = [c for c in self.columns_to_drop if c in X.columns]
        return self

    def transform(self, X):
        return X.drop(columns=self.to_drop_, errors="ignore")

class ConstantColumnDropper(BaseEstimator, TransformerMixin):
    """ Drops zero-variance columns. """
    def fit(self, X, y=None):
        self.to_drop_ = [col for col in X.columns if X[col].nunique(dropna=False) <= 1]
        return self
    
    def transform(self, X):
        return X.drop(columns=self.to_drop_, errors="ignore")

class HighMissingnessDropper(BaseEstimator, TransformerMixin):
    """ Drops columns with > threshold missing values. """
    def __init__(self, missingness_threshold=0.75):
        self.missingness_threshold = missingness_threshold

    def fit(self, X, y=None):
        missing_ratio = X.isna().mean()
        self.to_drop_ = missing_ratio[missing_ratio > self.missingness_threshold].index.tolist()
        return self

    def transform(self, X):
        return X.drop(columns=self.to_drop_, errors="ignore")

class HighCardinalityDropper(BaseEstimator, TransformerMixin):
    """ Drops categorical columns with too many unique values. """
    def __init__(self, max_unique_values=100):
        self.max_unique_values = max_unique_values

    def fit(self, X, y=None):
        self.to_drop_ = [
            col for col in X.select_dtypes(include="object").columns
            if X[col].nunique(dropna=False) > self.max_unique_values
        ]
        return self

    def transform(self, X):
        return X.drop(columns=self.to_drop_, errors="ignore")

class DateColumnDropper(BaseEstimator, TransformerMixin):
    """ Drops remaining datetime columns before ML. """
    def fit(self, X, y=None):
        self.to_drop_ = X.select_dtypes(include=["datetime64[ns]"]).columns.tolist()
        return self

    def transform(self, X):
        return X.drop(columns=self.to_drop_, errors="ignore")



if __name__ == "__main__":
    print("Transformer classes created successfully.")
