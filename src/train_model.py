import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, precision_recall_curve

# Custom preprocessing
from preprocess import (
    MissingValueNormalizer,
    TermParser,
    EmploymentLengthParser,
    ZipCodeParser,
    NumericCoercer,
    DateParser,
    DomainFeatureEngineer,
    DomainZeroImputer,
    ReferenceTimeImputer,
    PostEventFeatureDropper,
    ExplicitColumnDropper,
    ConstantColumnDropper,
    HighMissingnessDropper,
    HighCardinalityDropper,
    DateColumnDropper
)

# Configuration
DATA_PATH = "data/processed/lending_club_subset.csv"
TARGET_COL = "loan_status"
DATE_COL = "issue_d"
RANDOM_STATE = 42
TRAIN_TEST_SPLIT_DATE = "2018-01-01"

# Strict post-approval features (behavioral / outcome-driven)
POST_EVENT_COLS = [
    # repayment & recovery
    "out_prncp", "out_prncp_inv",
    "total_pymnt", "total_pymnt_inv",
    "total_rec_prncp", "total_rec_int",
    "total_rec_late_fee", "recoveries",
    "collection_recovery_fee",
    "last_pymnt_amnt",

    # post-issue dates
    "last_pymnt_d", "next_pymnt_d", "last_credit_pull_d",

    # updated credit scores
    "last_fico_range_high", "last_fico_range_low",

    # hardship
    "hardship_flag", "hardship_type", "hardship_reason",
    "hardship_status", "deferral_term", "hardship_amount",
    "hardship_start_date", "hardship_end_date",
    "payment_plan_start_date", "hardship_length",
    "hardship_dpd", "hardship_loan_status",
    "orig_projected_additional_accrued_interest",
    "hardship_payoff_balance_amount",
    "hardship_last_payment_amount",

    # settlement
    "debt_settlement_flag", "debt_settlement_flag_date",
    "settlement_status", "settlement_date",
    "settlement_amount", "settlement_percentage",
    "settlement_term"
]

# Explicit governance + proxy removal
EXPLICIT_DROP_COLS = [
    "id", "member_id", "url",
    "desc", "title", "grade",
    "sub_grade", "int_rate"
]

# Columns where missing means "never happened" (not missing data)
REFERENCE_TIME_COLS = [
    "mths_since_last_delinq",
    "mths_since_last_record",
    "mths_since_last_major_derog",
    "mths_since_recent_bc_dlq",
    "mths_since_recent_revol_delinq"
]

def main():
    """Main training function."""
    from pathlib import Path
    
    # Load data
    df = pd.read_csv(DATA_PATH)
    
    # Binary classification only
    df = df[df[TARGET_COL].isin(["Fully Paid", "Charged Off"])]
    
    y = (df[TARGET_COL] == "Charged Off").astype(int)
    X = df.drop(columns=[TARGET_COL])
    
    # Time-based split (Ensure data is sorted so TimeSeriesSplit respects the timeline)
    X[DATE_COL] = pd.to_datetime(X[DATE_COL])
    
    X = X.sort_values(by=DATE_COL)
    y = y.loc[X.index]
    
    train_mask = X[DATE_COL] < TRAIN_TEST_SPLIT_DATE
    test_mask  = X[DATE_COL] >= TRAIN_TEST_SPLIT_DATE
    
    X_train = X.loc[train_mask].copy()
    X_test  = X.loc[test_mask].copy()
    
    y_train = y.loc[train_mask]
    y_test  = y.loc[test_mask]
    
    print("Train shape:", X_train.shape)
    print("Test shape :", X_test.shape)
    
    # Calculate class weight for imbalanced data
    scale_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Scale pos weight: {scale_weight:.2f}")
    
    # Preprocessing pipeline (FROZEN & AUDITABLE)
    data_cleaning = Pipeline(steps=[
        ("missing_normalizer", MissingValueNormalizer()),
        ("term_parser", TermParser()),
        ("emp_length_parser", EmploymentLengthParser()),
        ("zip_parser", ZipCodeParser()),
        ("numeric_coercer", NumericCoercer()),
        ("date_parser", DateParser([DATE_COL, "earliest_cr_line"])),
        ("domain_features", DomainFeatureEngineer()),
        ("domain_imputer", DomainZeroImputer()),
        ("reference_time_imputer", ReferenceTimeImputer(REFERENCE_TIME_COLS, fill_value=999)),
        ("post_event_dropper", PostEventFeatureDropper(POST_EVENT_COLS)),
        ("explicit_dropper", ExplicitColumnDropper(EXPLICIT_DROP_COLS)),
        ("constant_dropper", ConstantColumnDropper()),
        ("missingness_dropper", HighMissingnessDropper(0.75)),
        ("cardinality_dropper", HighCardinalityDropper(100)),
        ("date_dropper", DateColumnDropper())
    ])
    
    # ML preprocessing
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])
    
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    ml_preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, make_column_selector(dtype_include=np.number)),
            ("cat", categorical_pipe, make_column_selector(dtype_include=object))
        ],
        remainder="drop"
    )
    
    # Full pipeline
    pipeline = Pipeline(steps=[
        ("data_cleaning", data_cleaning),
        ("ml_preprocessing", ml_preprocessor),
        ("model", XGBClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            scale_pos_weight=scale_weight,
            eval_metric="auc"
        ))
    ])
    
    # Hyperparameter tuning
    tscv = TimeSeriesSplit(n_splits=3)
    
    param_distributions = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [3, 6, 10],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__subsample": [0.7, 0.8, 1.0],
        "model__colsample_bytree": [0.7, 0.8, 1.0]
    }
    
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=20,               
        scoring="roc_auc",         
        cv=tscv,
        n_jobs=-1,
        verbose=2,
        random_state=RANDOM_STATE
    )
    
    print("\nRunning hyperparameter tuning...")
    search.fit(X_train, y_train)
    
    best_model = search.best_estimator_
    
    print("\nBest Parameters:")
    print(search.best_params_)
    
    # Final evaluation
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    print("\n--- Final Evaluation ---")
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("Precision: ", precision_score(y_test, y_pred))
    print("Recall: ", recall_score(y_test, y_pred))
    print("f1-score: ", f1_score(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # export pipeline
    model_path = Path("models/credit_risk_model.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_path)
    print(f"\nModel saved at {model_path}")


if __name__ == "__main__":
    main()