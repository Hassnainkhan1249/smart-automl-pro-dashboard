# Smart AutoML Pro Dashboard - Premium Version
# ------------------------------------------------------------
# Project Title:
# Smart AutoML Pro: Visual Automated Machine Learning Model Selector
#
# This version adds:
# - better target selection
# - data quality advisor
# - data cleaning advisor
# - class balance insight
# - correlation insight
# - model leaderboard
# - explainability
# - prediction form
# - exportable report
# ------------------------------------------------------------

import tempfile
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


# ------------------------------------------------------------
# Page Setup
# ------------------------------------------------------------

st.set_page_config(
    page_title="Smart AutoML Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------
# Professional CSS Styling
# ------------------------------------------------------------

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    .hero-box {
        background: linear-gradient(135deg, #101828 0%, #1D2939 55%, #344054 100%);
        padding: 30px 34px;
        border-radius: 24px;
        color: white;
        margin-bottom: 22px;
        box-shadow: 0px 12px 30px rgba(16, 24, 40, 0.18);
    }

    .hero-title {
        font-size: 42px;
        font-weight: 850;
        line-height: 1.05;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        font-size: 17px;
        color: #D0D5DD;
        max-width: 980px;
        line-height: 1.55;
    }

    .section-card {
        background: #FFFFFF;
        border: 1px solid #EAECF0;
        border-radius: 18px;
        padding: 22px;
        box-shadow: 0px 3px 14px rgba(16, 24, 40, 0.05);
        margin-bottom: 18px;
    }

    .soft-card {
        background: #F9FAFB;
        border: 1px solid #EAECF0;
        border-radius: 18px;
        padding: 18px;
        margin-bottom: 14px;
    }

    .winner-card {
        background: linear-gradient(135deg, #ECFDF3 0%, #D1FADF 100%);
        border: 1px solid #ABEFC6;
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0px 6px 18px rgba(18, 183, 106, 0.12);
        margin-bottom: 18px;
    }

    .risk-card {
        background: #FFF6ED;
        border: 1px solid #FEDF89;
        border-radius: 18px;
        padding: 18px;
        margin-bottom: 14px;
    }

    .bad-card {
        background: #FEF3F2;
        border: 1px solid #FECDCA;
        border-radius: 18px;
        padding: 18px;
        margin-bottom: 14px;
    }

    .info-card {
        background: #EFF8FF;
        border: 1px solid #B2DDFF;
        border-radius: 18px;
        padding: 18px;
        margin-bottom: 14px;
    }

    .mini-title {
        font-size: 15px;
        font-weight: 700;
        color: #344054;
        margin-bottom: 6px;
    }

    .mini-text {
        color: #667085;
        font-size: 14px;
        line-height: 1.45;
    }

    .pipeline-step {
        background: #F8FAFC;
        border: 1px solid #EAECF0;
        border-radius: 16px;
        padding: 16px 12px;
        min-height: 92px;
        text-align: center;
        font-weight: 700;
        color: #344054;
        box-shadow: 0px 2px 8px rgba(16,24,40,0.04);
    }

    .small-muted {
        color: #667085;
        font-size: 13px;
    }

    div[data-testid="stMetricValue"] {
        font-size: 27px;
        font-weight: 800;
        color: #101828;
    }

    div[data-testid="stMetricLabel"] {
        color: #475467;
        font-weight: 650;
    }

    .stButton>button {
        border-radius: 12px;
        font-weight: 750;
        padding: 0.6rem 1rem;
    }

    .stDownloadButton>button {
        border-radius: 12px;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# Demo Dataset Generator
# ------------------------------------------------------------

@st.cache_data
def generate_demo_dataset(rows: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Creates a realistic synthetic cancer-awareness dataset.
    This helps test the dashboard without needing external data.
    """
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 45, rows)
    gender = rng.choice(["Male", "Female"], rows, p=[0.48, 0.52])
    area = rng.choice(["Urban", "Semi-Urban", "Rural"], rows, p=[0.45, 0.25, 0.30])
    education = rng.choice(
        ["Matric", "Intermediate", "Undergraduate", "Graduate"],
        rows,
        p=[0.18, 0.30, 0.36, 0.16],
    )
    income = rng.choice(["Low", "Middle", "High"], rows, p=[0.38, 0.48, 0.14])
    family_history = rng.choice(["Yes", "No"], rows, p=[0.22, 0.78])
    smoking = rng.choice(["Never", "Former", "Current"], rows, p=[0.62, 0.15, 0.23])
    internet_use = rng.choice(["Low", "Moderate", "High"], rows, p=[0.25, 0.42, 0.33])
    attended_session = rng.choice(["Yes", "No"], rows, p=[0.34, 0.66])
    knows_screening = rng.choice(["Yes", "No"], rows, p=[0.46, 0.54])
    heard_cancer = rng.choice(["Yes", "No"], rows, p=[0.82, 0.18])

    score = (
        35
        + np.where(education == "Graduate", 16, 0)
        + np.where(education == "Undergraduate", 12, 0)
        + np.where(education == "Intermediate", 6, 0)
        + np.where(area == "Urban", 7, 0)
        + np.where(area == "Semi-Urban", 4, 0)
        + np.where(internet_use == "High", 9, 0)
        + np.where(internet_use == "Moderate", 5, 0)
        + np.where(attended_session == "Yes", 14, 0)
        + np.where(knows_screening == "Yes", 16, 0)
        + np.where(heard_cancer == "Yes", 8, 0)
        + np.where(family_history == "Yes", 5, 0)
        - np.where(smoking == "Current", 3, 0)
        + rng.normal(0, 7, rows)
    )

    score = np.clip(score, 0, 100).round(1)

    awareness_level = pd.cut(
        score,
        bins=[-1, 49, 74, 101],
        labels=["Low", "Medium", "High"],
    ).astype(str)

    df = pd.DataFrame(
        {
            "Student_ID": [f"STU-{1000 + i}" for i in range(rows)],
            "Age": age,
            "Gender": gender,
            "Area": area,
            "Education_Level": education,
            "Income_Level": income,
            "Family_Cancer_History": family_history,
            "Smoking_Status": smoking,
            "Internet_Use": internet_use,
            "Attended_Health_Session": attended_session,
            "Knows_Cancer_Screening": knows_screening,
            "Heard_About_Cancer": heard_cancer,
            "Awareness_Score": score,
            "Awareness_Level": awareness_level,
        }
    )

    # Controlled missing values to test automatic preprocessing.
    missing_columns = [
        "Education_Level",
        "Income_Level",
        "Internet_Use",
        "Knows_Cancer_Screening",
        "Age",
    ]

    for col in missing_columns:
        missing_idx = rng.choice(rows, size=int(rows * 0.025), replace=False)
        df.loc[missing_idx, col] = np.nan

    return df


# ------------------------------------------------------------
# ID Detection and Target Recommendation
# ------------------------------------------------------------

def is_id_like_column(df: pd.DataFrame, col: str) -> bool:
    """
    Detects true ID/serial-like columns.
    This function is intentionally conservative so columns like 'diagnosis'
    are not wrongly removed.
    """
    col_lower = str(col).lower()
    unique_ratio = df[col].nunique(dropna=True) / max(len(df), 1)

    strong_exact_id_names = [
        "id",
        "student_id",
        "record_id",
        "patient_id",
        "serial",
        "serial_no",
        "sr_no",
        "roll_no",
        "registration",
        "registration_no",
        "cnic",
        "email",
        "phone",
    ]

    if col_lower in strong_exact_id_names:
        return True

    if col_lower.endswith("_id"):
        return True

    if unique_ratio >= 0.98 and pd.api.types.is_numeric_dtype(df[col]):
        return True

    if unique_ratio >= 0.98 and col_lower in ["name", "email", "phone", "cnic"]:
        return True

    return False


def recommend_target_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scores each column to suggest whether it is a good target column.
    Higher score = better target candidate.
    """
    rows = []

    preferred_keywords = [
        "target",
        "label",
        "class",
        "level",
        "status",
        "result",
        "outcome",
        "score",
        "grade",
        "diagnosis",
        "prediction",
        "awareness",
        "risk",
        "disease",
    ]

    for col in df.columns:
        col_lower = str(col).lower()
        unique_count = df[col].nunique(dropna=True)
        unique_ratio = unique_count / max(len(df), 1)
        missing_ratio = df[col].isna().mean()
        numeric = pd.api.types.is_numeric_dtype(df[col])
        id_like = is_id_like_column(df, col)

        score = 50
        reasons = []

        if any(keyword in col_lower for keyword in preferred_keywords):
            score += 35
            reasons.append("Name suggests outcome/label")

        if id_like:
            score -= 90
            reasons.append("Looks like ID/identifier")

        if missing_ratio > 0.30:
            score -= 25
            reasons.append("High missing values")

        if unique_count <= 1:
            score -= 90
            reasons.append("Only one unique value")

        elif numeric and 10 < unique_count < len(df) * 0.90:
            score += 12
            reasons.append("Suitable numeric prediction target")

        elif not numeric and 2 <= unique_count <= 25:
            score += 18
            reasons.append("Suitable classification target")

        elif unique_ratio >= 0.90:
            score -= 35
            reasons.append("Too many unique values")

        score = max(0, min(100, score))

        if id_like:
            recommendation = "Do not use"
        elif score >= 80:
            recommendation = "Highly recommended"
        elif score >= 60:
            recommendation = "Possible"
        else:
            recommendation = "Weak choice"

        rows.append(
            {
                "Column": col,
                "Recommendation": recommendation,
                "Target Score": score,
                "Unique Values": unique_count,
                "Missing %": round(missing_ratio * 100, 2),
                "Data Type": str(df[col].dtype),
                "Reason": "; ".join(reasons) if reasons else "General column",
            }
        )

    return pd.DataFrame(rows).sort_values(
        by="Target Score", ascending=False
    ).reset_index(drop=True)


# ------------------------------------------------------------
# Dataset Quality and Cleaning Advisor
# ------------------------------------------------------------

def dataset_quality(df: pd.DataFrame) -> tuple[float, str, dict]:
    """
    Calculates a simple dataset quality score out of 100.
    The score is based on missing values, duplicates, and ID-like columns.
    """
    total_cells = max(df.shape[0] * df.shape[1], 1)

    missing_ratio = df.isna().sum().sum() / total_cells
    duplicate_ratio = df.duplicated().sum() / max(df.shape[0], 1)
    id_cols = [col for col in df.columns if is_id_like_column(df, col)]

    score = 100
    score -= missing_ratio * 45
    score -= duplicate_ratio * 35
    score -= min(len(id_cols), 5) * 1.5
    score = round(max(0, min(100, score)), 2)

    if score >= 90:
        label = "Excellent"
    elif score >= 75:
        label = "Good"
    elif score >= 55:
        label = "Moderate"
    else:
        label = "Needs cleaning"

    details = {
        "missing_ratio": missing_ratio,
        "duplicate_ratio": duplicate_ratio,
        "id_columns": id_cols,
    }

    return score, label, details


def create_cleaning_advisor(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gives practical cleaning suggestions for each column.
    """
    advice_rows = []

    for col in df.columns:
        missing_percent = df[col].isna().mean() * 100
        unique_count = df[col].nunique(dropna=True)
        unique_ratio = unique_count / max(len(df), 1)
        dtype = str(df[col].dtype)
        id_like = is_id_like_column(df, col)

        issues = []
        suggestion = "No major issue detected."

        if id_like:
            issues.append("ID-like column")
            suggestion = "Exclude from model features unless it has real predictive meaning."

        if missing_percent > 40:
            issues.append("Very high missing values")
            suggestion = "Consider dropping this column or collecting cleaner data."

        elif missing_percent > 10:
            issues.append("Moderate missing values")
            suggestion = "Use imputation and review data quality."

        if unique_count <= 1:
            issues.append("Constant column")
            suggestion = "Drop this column because it adds no useful information."

        if dtype == "object" and unique_ratio > 0.60 and not id_like:
            issues.append("High-cardinality categorical column")
            suggestion = "Review before modeling; may need grouping or removal."

        advice_rows.append(
            {
                "Column": col,
                "Data Type": dtype,
                "Missing %": round(missing_percent, 2),
                "Unique Values": unique_count,
                "Issue": ", ".join(issues) if issues else "None",
                "Suggested Action": suggestion,
            }
        )

    return pd.DataFrame(advice_rows)


def detect_problem_type(y: pd.Series) -> str:
    """
    Detects classification or regression based on the selected target.
    """
    y_clean = y.dropna()

    if (
        pd.api.types.is_bool_dtype(y_clean)
        or pd.api.types.is_object_dtype(y_clean)
        or str(y_clean.dtype) == "category"
    ):
        return "classification"

    unique_count = y_clean.nunique(dropna=True)
    unique_ratio = unique_count / max(len(y_clean), 1)

    if unique_count <= 12 or unique_ratio < 0.05:
        return "classification"

    return "regression"


def validate_target(df: pd.DataFrame, target_col: str) -> tuple[bool, str]:
    """
    Checks whether selected target column is meaningful.
    """
    if is_id_like_column(df, target_col):
        return (
            False,
            "This column looks like an ID or unique identifier. Select a real outcome column instead.",
        )

    y = df[target_col].dropna()

    if y.nunique() < 2:
        return (
            False,
            "The selected target has fewer than two unique values. It cannot be predicted meaningfully.",
        )

    if df[target_col].isna().mean() > 0.35:
        return (
            False,
            "The selected target has too many missing values. Choose a cleaner target column.",
        )

    return True, "Target looks usable."


def prepare_xy(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series, list]:
    """
    Prepares input features X and target y.
    Removes ID-like feature columns automatically.
    """
    drop_cols = [target_col]
    id_features = []

    for col in df.columns:
        if col != target_col and is_id_like_column(df, col):
            drop_cols.append(col)
            id_features.append(col)

    clean_df = df.copy()
    clean_df = clean_df.dropna(subset=[target_col])

    X = clean_df.drop(columns=drop_cols)
    y = clean_df[target_col]

    return X, y, id_features

# ------------------------------------------------------------
# Preprocessing and Model Setup
# ------------------------------------------------------------

def make_onehot_encoder():
    """
    Creates OneHotEncoder compatible with both older and newer
    scikit-learn versions.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Builds preprocessing pipeline.

    Numeric columns:
    - median imputation
    - standard scaling

    Categorical columns:
    - most frequent imputation
    - one-hot encoding
    """
    numeric_features = X.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", make_onehot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor


def get_candidate_models(problem_type: str):
    """
    Returns candidate models based on whether the task is
    classification or regression.
    """
    if problem_type == "classification":
        return {
            "Logistic Regression": LogisticRegression(max_iter=2000),
            "KNN Classifier": KNeighborsClassifier(n_neighbors=7),
            "Decision Tree": DecisionTreeClassifier(
                random_state=42,
                max_depth=8,
            ),
            "Random Forest": RandomForestClassifier(
                random_state=42,
                n_estimators=150,
                max_depth=None,
            ),
            "SVM": SVC(
                probability=True,
                kernel="rbf",
            ),
            "Naive Bayes": GaussianNB(),
            "Gradient Boosting": GradientBoostingClassifier(
                random_state=42,
            ),
        }

    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Polynomial Regression": Pipeline(
            steps=[
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("linear", Ridge(alpha=1.0)),
            ]
        ),
        "KNN Regressor": KNeighborsRegressor(n_neighbors=7),
        "Decision Tree Regressor": DecisionTreeRegressor(
            random_state=42,
            max_depth=8,
        ),
        "Random Forest Regressor": RandomForestRegressor(
            random_state=42,
            n_estimators=150,
        ),
        "SVR": SVR(kernel="rbf"),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            random_state=42,
        ),
    }


# ------------------------------------------------------------
# Training and Evaluation
# ------------------------------------------------------------

def train_models(
    X: pd.DataFrame,
    y: pd.Series,
    problem_type: str,
    test_size: float,
    cv_folds: int,
):
    """
    Trains multiple models and returns:
    - leaderboard
    - best model name
    - best model information
    - all trained pipelines
    """
    models = get_candidate_models(problem_type)
    preprocessor = build_preprocessor(X)

    stratify_value = None

    if problem_type == "classification":
        class_counts = y.value_counts()

        if len(class_counts) >= 2 and class_counts.min() >= 2:
            stratify_value = y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=stratify_value,
    )

    results = []
    trained_models = {}

    progress_bar = st.progress(0)
    status_box = st.empty()

    for index, (model_name, model) in enumerate(models.items(), start=1):
        status_box.info(f"Training model: {model_name}")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        try:
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            cv_mean = np.nan

            try:
                scoring = "f1_weighted" if problem_type == "classification" else "r2"

                safe_cv = min(cv_folds, max(2, len(X_train) // 20))

                if problem_type == "classification":
                    min_class_count = y_train.value_counts().min()
                    safe_cv = min(safe_cv, int(min_class_count))

                if safe_cv >= 2:
                    cv_scores = cross_val_score(
                        pipeline,
                        X_train,
                        y_train,
                        cv=safe_cv,
                        scoring=scoring,
                    )
                    cv_mean = float(np.mean(cv_scores))

            except Exception:
                cv_mean = np.nan

            if problem_type == "classification":
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(
                    y_test,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                )
                recall = recall_score(
                    y_test,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                )
                f1 = f1_score(
                    y_test,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                )

                results.append(
                    {
                        "Model": model_name,
                        "Accuracy": accuracy,
                        "Precision": precision,
                        "Recall": recall,
                        "F1 Score": f1,
                        "CV Score": cv_mean,
                        "Main Score": f1,
                        "Status": "Success",
                        "Error": "",
                    }
                )

            else:
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)

                results.append(
                    {
                        "Model": model_name,
                        "MAE": mae,
                        "RMSE": rmse,
                        "R2 Score": r2,
                        "CV Score": cv_mean,
                        "Main Score": r2,
                        "Status": "Success",
                        "Error": "",
                    }
                )

            trained_models[model_name] = {
                "pipeline": pipeline,
                "y_test": y_test,
                "y_pred": y_pred,
                "X_test": X_test,
            }

        except Exception as exc:
            results.append(
                {
                    "Model": model_name,
                    "Main Score": -999,
                    "CV Score": np.nan,
                    "Status": "Failed",
                    "Error": str(exc),
                }
            )

        progress_bar.progress(index / len(models))

    status_box.success("Model training completed.")

    leaderboard = pd.DataFrame(results)
    leaderboard = leaderboard.sort_values(
        by="Main Score",
        ascending=False,
    ).reset_index(drop=True)

    successful_models = leaderboard[leaderboard["Status"] == "Success"]

    if successful_models.empty:
        raise RuntimeError(
            "No model could be trained successfully. Check dataset and target column."
        )

    best_model_name = successful_models.iloc[0]["Model"]
    best_model_info = trained_models[best_model_name]

    return leaderboard, best_model_name, best_model_info, trained_models


# ------------------------------------------------------------
# Model Explanation
# ------------------------------------------------------------

def model_explanation(model_name: str, problem_type: str) -> str:
    """
    Gives simple explanation of why a selected model may perform well.
    """
    explanations = {
        "Logistic Regression": (
            "Logistic Regression is a strong baseline for classification. "
            "It performs well when the relationship between input features "
            "and classes is relatively simple."
        ),
        "KNN Classifier": (
            "KNN predicts a class by comparing a new case with the most similar "
            "examples in the dataset."
        ),
        "Decision Tree": (
            "Decision Tree creates rule-based splits in the data. It is easy to "
            "interpret but may overfit if not controlled."
        ),
        "Random Forest": (
            "Random Forest combines many decision trees. It usually performs well "
            "on mixed numeric and categorical datasets and handles nonlinear patterns."
        ),
        "SVM": (
            "SVM tries to find a strong boundary between classes. It can perform well "
            "on medium-sized structured datasets."
        ),
        "Naive Bayes": (
            "Naive Bayes is a fast probability-based classifier. It is useful as a "
            "simple benchmark model."
        ),
        "Gradient Boosting": (
            "Gradient Boosting builds models step by step, where each new model tries "
            "to correct the mistakes of the previous models."
        ),
        "Linear Regression": (
            "Linear Regression predicts numeric values using a straight-line relationship "
            "between input features and the target."
        ),
        "Ridge Regression": (
            "Ridge Regression is a more stable form of linear regression. It reduces "
            "overfitting by applying regularization."
        ),
        "Polynomial Regression": (
            "Polynomial Regression captures curved relationships by creating polynomial "
            "feature combinations."
        ),
        "KNN Regressor": (
            "KNN Regressor predicts numeric values by averaging the values of similar "
            "nearby records."
        ),
        "Decision Tree Regressor": (
            "Decision Tree Regressor predicts numeric values using rule-based splits "
            "in the data."
        ),
        "Random Forest Regressor": (
            "Random Forest Regressor combines many regression trees and handles nonlinear "
            "relationships effectively."
        ),
        "SVR": (
            "SVR predicts continuous values using support vector logic and controls "
            "prediction error within a margin."
        ),
        "Gradient Boosting Regressor": (
            "Gradient Boosting Regressor improves numeric predictions step by step by "
            "correcting previous errors."
        ),
    }

    return explanations.get(
        model_name,
        "This model achieved the strongest score among the tested candidates.",
    )


def generate_model_insight(
    leaderboard: pd.DataFrame,
    best_model_name: str,
    problem_type: str,
) -> str:
    """
    Generates a short insight from the leaderboard.
    """
    successful = leaderboard[leaderboard["Status"] == "Success"].copy()

    if successful.empty:
        return "No successful model results are available."

    best_score = float(successful.iloc[0]["Main Score"])

    if len(successful) > 1:
        second_score = float(successful.iloc[1]["Main Score"])
        margin = best_score - second_score
    else:
        margin = 0

    if problem_type == "classification":
        metric_name = "Weighted F1 Score"
    else:
        metric_name = "R² Score"

    if margin > 0.10:
        strength = "The winning model performed clearly better than the next model."
    elif margin > 0.03:
        strength = "The winning model performed moderately better than the next model."
    else:
        strength = (
            "The top models performed closely, so the final choice should also consider "
            "interpretability and stability."
        )

    return (
        f"{best_model_name} achieved the highest {metric_name} of {best_score:.4f}. "
        f"{strength}"
    )


def create_feature_importance(best_model_info) -> pd.DataFrame | None:
    """
    Extracts feature importance for tree-based models.
    Returns None if the selected model does not support feature_importances_.
    """
    try:
        pipeline = best_model_info["pipeline"]
        model = pipeline.named_steps["model"]

        if not hasattr(model, "feature_importances_"):
            return None

        preprocessor = pipeline.named_steps["preprocessor"]
        feature_names = preprocessor.get_feature_names_out()
        importances = model.feature_importances_

        importance_df = pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": importances,
            }
        )

        importance_df = importance_df.sort_values(
            by="Importance",
            ascending=False,
        ).head(15)

        return importance_df.reset_index(drop=True)

    except Exception:
        return None


# ------------------------------------------------------------
# Data Insight Charts
# ------------------------------------------------------------

def create_missing_values_chart(df: pd.DataFrame):
    """
    Creates a Plotly bar chart for missing values.
    """
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        return None

    missing_df = pd.DataFrame(
        {
            "Column": missing.index,
            "Missing Values": missing.values,
        }
    )

    fig = px.bar(
        missing_df,
        x="Missing Values",
        y="Column",
        orientation="h",
        title="Missing Values by Column",
    )

    fig.update_layout(
        height=420,
        margin=dict(l=10, r=20, t=50, b=20),
    )

    return fig


def create_column_type_chart(df: pd.DataFrame):
    """
    Creates a donut chart showing numeric vs categorical columns.
    """
    numeric_count = len(
        df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns
    )

    categorical_count = len(
        df.select_dtypes(include=["object", "category", "bool"]).columns
    )

    chart_df = pd.DataFrame(
        {
            "Type": ["Numeric", "Categorical/Boolean"],
            "Columns": [numeric_count, categorical_count],
        }
    )

    fig = px.pie(
        chart_df,
        values="Columns",
        names="Type",
        hole=0.45,
        title="Column Type Distribution",
    )

    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=50, b=10),
    )

    return fig


def create_class_balance_chart(y: pd.Series):
    """
    Creates a class balance chart for classification targets.
    """
    counts = y.value_counts(dropna=False).reset_index()
    counts.columns = ["Class", "Count"]

    fig = px.bar(
        counts,
        x="Class",
        y="Count",
        title="Target Class Balance",
        text="Count",
    )

    fig.update_layout(
        height=380,
        margin=dict(l=10, r=20, t=50, b=20),
    )

    return fig


def create_numeric_distribution_chart(df: pd.DataFrame, column: str):
    """
    Creates a histogram for a numeric column.
    """
    fig = px.histogram(
        df,
        x=column,
        nbins=30,
        title=f"Distribution of {column}",
    )

    fig.update_layout(
        height=380,
        margin=dict(l=10, r=20, t=50, b=20),
    )

    return fig


def create_correlation_heatmap(df: pd.DataFrame):
    """
    Creates a numeric correlation heatmap.
    """
    numeric_df = df.select_dtypes(include=["int64", "float64", "int32", "float32"])

    if numeric_df.shape[1] < 2:
        return None

    corr = numeric_df.corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        title="Numeric Feature Correlation Heatmap",
    )

    fig.update_layout(
        height=580,
        margin=dict(l=10, r=20, t=50, b=20),
    )

    return fig


def create_target_correlation_chart(df: pd.DataFrame, target_col: str):
    """
    Creates a chart showing top numeric correlations with a numeric target.
    """
    if target_col not in df.columns:
        return None

    numeric_df = df.select_dtypes(include=["int64", "float64", "int32", "float32"])

    if target_col not in numeric_df.columns:
        return None

    if numeric_df.shape[1] < 2:
        return None

    corr = numeric_df.corr()[target_col]
    corr = corr.drop(labels=[target_col], errors="ignore")
    corr = corr.dropna()
    corr = corr.reindex(corr.abs().sort_values(ascending=False).index)
    corr = corr.head(12)

    if corr.empty:
        return None

    corr_df = pd.DataFrame(
        {
            "Feature": corr.index,
            "Correlation": corr.values,
        }
    )

    fig = px.bar(
        corr_df.sort_values("Correlation"),
        x="Correlation",
        y="Feature",
        orientation="h",
        title=f"Top Numeric Correlations with {target_col}",
    )

    fig.update_layout(
        height=430,
        margin=dict(l=10, r=20, t=50, b=20),
    )

    return fig


# ------------------------------------------------------------
# Report and Export Utilities
# ------------------------------------------------------------

def create_report(
    df: pd.DataFrame,
    target_col: str,
    problem_type: str,
    leaderboard: pd.DataFrame,
    best_model_name: str,
    best_score: float,
    quality_score: float,
    quality_label: str,
    id_features: list,
    cleaning_advice: pd.DataFrame,
) -> str:
    """
    Creates an exportable text report.
    """
    successful = leaderboard[leaderboard["Status"] == "Success"]
    tested_models = ", ".join(leaderboard["Model"].astype(str).tolist())

    if problem_type == "classification":
        metric_line = "Main selection metric: Weighted F1 Score"
    else:
        metric_line = "Main selection metric: R² Score"

    major_issues = cleaning_advice[cleaning_advice["Issue"] != "None"]

    if major_issues.empty:
        issue_summary = "No major data quality issues detected."
    else:
        issue_summary = "; ".join(
            major_issues.head(5)["Column"].astype(str).tolist()
        )

    report = f"""
SMART AUTOML PRO DASHBOARD - AUTO-GENERATED PROJECT REPORT
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}

1. PROJECT PURPOSE
This project demonstrates a visual AutoML system that automatically analyzes a dataset, recommends a target, detects the machine learning task, trains multiple algorithms, compares model performance, and recommends the best-performing model.

2. DATASET SUMMARY
- Rows: {df.shape[0]}
- Columns: {df.shape[1]}
- Target Column: {target_col}
- Missing Values: {int(df.isna().sum().sum())}
- Duplicate Rows: {int(df.duplicated().sum())}
- Dataset Quality Score: {quality_score}/100 ({quality_label})
- ID-like columns excluded from features: {', '.join(id_features) if id_features else 'None'}

3. DATA CLEANING ADVISOR SUMMARY
- Key columns/issues to review: {issue_summary}
- The system automatically handles standard missing values and categorical encoding during model training.

4. DETECTED MACHINE LEARNING TASK
- Problem Type: {problem_type.title()}
- Detection Method: The system checked the selected target column data type, number of unique values, and target structure.
- {metric_line}

5. PREPROCESSING PIPELINE
- Missing numeric values handled using median imputation.
- Missing categorical values handled using most frequent value imputation.
- Numeric variables scaled using StandardScaler.
- Categorical variables encoded using One-Hot Encoding.
- ID-like columns removed from model features.
- Dataset split into training and testing sets.

6. MODELS TESTED
{tested_models}

7. BEST MODEL SELECTED
- Best Model: {best_model_name}
- Best Score: {round(float(best_score), 4)}
- Explanation: {model_explanation(best_model_name, problem_type)}

8. MODEL LEADERBOARD SUMMARY
Successful models trained: {len(successful)} out of {len(leaderboard)}.
Model insight: {generate_model_insight(leaderboard, best_model_name, problem_type)}

9. CONCLUSION
The dashboard successfully demonstrates an end-to-end automated machine learning workflow. It is more advanced than a single-model prediction project because it includes dataset analysis, target guidance, preprocessing, model comparison, visual evaluation, explainability, prediction, and export within one interactive system.
"""
    return report.strip()


def dataframe_download(df: pd.DataFrame) -> bytes:
    """
    Converts dataframe to downloadable CSV bytes.
    """
    return df.to_csv(index=False).encode("utf-8")


def download_model_bytes(pipeline) -> bytes:
    """
    Converts trained model pipeline into downloadable .pkl bytes.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
        joblib.dump(pipeline, tmp.name)

        with open(tmp.name, "rb") as file:
            return file.read()

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------

st.markdown(
    """
    <div class="hero-box">
        <div class="hero-title">Smart AutoML Pro Dashboard</div>
        <div class="hero-subtitle">
            A visual automated machine learning dashboard that analyzes datasets,
            recommends target columns, trains multiple models, compares performance,
            explains the best model, and exports results.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# Sidebar Controls
# ------------------------------------------------------------

st.sidebar.title("Control Panel")
st.sidebar.caption("Upload a dataset or use the built-in demo dataset.")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file",
    type=["csv"],
)

use_demo = st.sidebar.toggle(
    "Use demo cancer-awareness dataset",
    value=False,
)

st.sidebar.divider()

st.sidebar.subheader("Training Settings")

test_size = st.sidebar.slider(
    "Test data size",
    min_value=0.10,
    max_value=0.40,
    value=0.20,
    step=0.05,
)

cv_folds = st.sidebar.slider(
    "Cross-validation folds",
    min_value=3,
    max_value=10,
    value=5,
    step=1,
)

st.sidebar.divider()

st.sidebar.subheader("Best Demo Targets")
st.sidebar.success("Awareness_Level = classification")
st.sidebar.info("Awareness_Score = regression")


# ------------------------------------------------------------
# Data Loading
# ------------------------------------------------------------

if use_demo:
    df = generate_demo_dataset()
    data_source_label = "Built-in cancer-awareness demo dataset"

elif uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        data_source_label = uploaded_file.name
    except Exception as error:
        st.error(f"Could not read CSV file. Error: {error}")
        st.stop()

else:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Start Here")
    st.write(
        "Upload a CSV dataset from the sidebar or turn on the built-in demo dataset. "
        "The system will then guide you through target selection, model training, "
        "result comparison, prediction, and export."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("System Workflow")

    workflow_cols = st.columns(7)

    workflow_steps = [
        "1. Load Data",
        "2. Recommend Target",
        "3. Detect Task",
        "4. Preprocess",
        "5. Train Models",
        "6. Compare Results",
        "7. Export Report",
    ]

    for col, step in zip(workflow_cols, workflow_steps):
        with col:
            st.markdown(
                f"<div class='pipeline-step'>{step}</div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    st.subheader("What this project demonstrates")

    d1, d2, d3, d4 = st.columns(4)

    with d1:
        st.markdown(
            """
            <div class='soft-card'>
                <div class='mini-title'>AutoML Logic</div>
                <div class='mini-text'>Automatically tests multiple machine learning models.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with d2:
        st.markdown(
            """
            <div class='soft-card'>
                <div class='mini-title'>Target Advisor</div>
                <div class='mini-text'>Guides the user to choose a meaningful prediction target.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with d3:
        st.markdown(
            """
            <div class='soft-card'>
                <div class='mini-title'>Model Comparison</div>
                <div class='mini-text'>Compares different algorithms through scores and visuals.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with d4:
        st.markdown(
            """
            <div class='soft-card'>
                <div class='mini-title'>Exportable Results</div>
                <div class='mini-text'>Exports reports, leaderboard, and trained model files.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.stop()


# ------------------------------------------------------------
# Initial Dataset Checks
# ------------------------------------------------------------

if df.empty:
    st.error("The uploaded dataset is empty.")
    st.stop()

# Remove unnamed index-like columns created by CSV exports.
unnamed_columns = [
    col for col in df.columns
    if str(col).lower().startswith("unnamed")
]

if unnamed_columns:
    df = df.drop(columns=unnamed_columns)

quality_score, quality_label, quality_details = dataset_quality(df)
target_recommendations = recommend_target_columns(df)
cleaning_advice = create_cleaning_advisor(df)

ordered_target_options = target_recommendations["Column"].tolist()

# Prefer known useful targets where present.
default_target_index = 0

for preferred_target in [
    "Awareness_Level",
    "diagnosis",
    "Diagnosis",
    "Outcome",
    "outcome",
    "Target",
    "target",
    "Class",
    "class",
    "Awareness_Score",
]:
    if preferred_target in ordered_target_options:
        default_target_index = ordered_target_options.index(preferred_target)
        break


# ------------------------------------------------------------
# Main Target Selection Panel
# ------------------------------------------------------------

st.markdown("<div class='section-card'>", unsafe_allow_html=True)

st.subheader("Choose Target Column")
st.write(
    "Select the column that the machine learning model should predict. "
    "A good target is usually an outcome such as diagnosis, class, level, status, score, or risk."
)

target_select_col, target_guide_col = st.columns([1.2, 1])

with target_select_col:
    target_col = st.selectbox(
        "Target column / output variable",
        ordered_target_options,
        index=default_target_index,
        help=(
            "Example: use diagnosis for breast cancer classification, "
            "Awareness_Level for awareness classification, or Awareness_Score "
            "for numeric prediction."
        ),
    )

with target_guide_col:
    st.info(
        "Avoid selecting ID columns, names, serial numbers, or ordinary input "
        "features unless you specifically want to predict them."
    )

st.markdown("</div>", unsafe_allow_html=True)

st.sidebar.divider()
st.sidebar.subheader("Selected Target")
st.sidebar.success(target_col)


# ------------------------------------------------------------
# Target Validation and Feature Preparation
# ------------------------------------------------------------

is_valid_target, target_message = validate_target(df, target_col)

if is_valid_target:
    problem_type = detect_problem_type(df[target_col])
    X, y, id_features = prepare_xy(df, target_col)
else:
    problem_type = "invalid"
    X = pd.DataFrame()
    y = pd.Series(dtype=float)
    id_features = []


# ------------------------------------------------------------
# Main Tabs
# ------------------------------------------------------------

tab_overview, tab_advisor, tab_training, tab_results, tab_predict, tab_export = st.tabs(
    [
        "1. Overview",
        "2. Advisor",
        "3. Train Models",
        "4. Results",
        "5. Predict",
        "6. Export",
    ]
)


# ------------------------------------------------------------
# Tab 1: Overview
# ------------------------------------------------------------

with tab_overview:
    st.header("Dataset Overview")
    st.caption(f"Data source: {data_source_label}")

    metric_1, metric_2, metric_3, metric_4, metric_5 = st.columns(5)

    metric_1.metric("Rows", f"{df.shape[0]:,}")
    metric_2.metric("Columns", f"{df.shape[1]:,}")
    metric_3.metric("Missing Values", f"{int(df.isna().sum().sum()):,}")
    metric_4.metric("Duplicates", f"{int(df.duplicated().sum()):,}")
    metric_5.metric("Quality", f"{quality_score}/100")

    if quality_score >= 90:
        st.success(f"Dataset quality is {quality_label}. It is suitable for model testing.")
    elif quality_score >= 75:
        st.info(f"Dataset quality is {quality_label}. It can be used, but check missing values.")
    elif quality_score >= 55:
        st.warning(
            f"Dataset quality is {quality_label}. Review missing values and weak columns before training."
        )
    else:
        st.error(
            f"Dataset quality is {quality_label}. Clean the dataset before serious model training."
        )

    st.subheader("Dataset Preview")
    st.dataframe(
        df.head(25),
        use_container_width=True,
    )

    st.subheader("Column Summary")

    column_summary = pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Missing Values": df.isna().sum().values,
            "Unique Values": df.nunique(dropna=True).values,
            "ID-like": [
                "Yes" if is_id_like_column(df, col) else "No"
                for col in df.columns
            ],
        }
    )

    st.dataframe(
        column_summary,
        use_container_width=True,
    )

    st.subheader("Data Quality Snapshot")

    overview_left, overview_right = st.columns([1, 1])

    with overview_left:
        fig_types = create_column_type_chart(df)
        st.plotly_chart(
            fig_types,
            use_container_width=True,
            key="overview_column_type_chart",
        )

    with overview_right:
        fig_missing = create_missing_values_chart(df)

        if fig_missing is not None:
            st.plotly_chart(
                fig_missing,
                use_container_width=True,
                key="overview_missing_values_chart",
            )
        else:
            st.success("No missing values detected in the dataset.")


# ------------------------------------------------------------
# Tab 2: Advisor
# ------------------------------------------------------------

with tab_advisor:
    st.header("Target and Data Advisor")

    selected_target_row = target_recommendations[
        target_recommendations["Column"] == target_col
    ].iloc[0]

    advisor_col_1, advisor_col_2, advisor_col_3 = st.columns(3)

    advisor_col_1.metric("Selected Target", target_col)
    advisor_col_2.metric("Target Status", "Valid" if is_valid_target else "Invalid")
    advisor_col_3.metric(
        "Detected Task",
        problem_type.title() if is_valid_target else "Invalid",
    )

    if is_valid_target:
        st.success(target_message)
    else:
        st.error(target_message)

    recommendation = selected_target_row["Recommendation"]

    if recommendation == "Highly recommended":
        st.success(f"{target_col} is highly recommended as a target column.")
    elif recommendation == "Possible":
        st.info(
            f"{target_col} is possible, but confirm that it is truly the outcome you want to predict."
        )
    elif recommendation == "Weak choice":
        st.warning(
            f"{target_col} is a weak target choice. Prefer a real outcome such as diagnosis, outcome, level, status, class, or score."
        )
    else:
        st.error(f"{target_col} should not be used as a target column.")

    st.subheader("Target Column Recommendations")

    st.dataframe(
        target_recommendations,
        use_container_width=True,
    )

    st.subheader("Data Cleaning Advisor")

    st.write(
        "This table identifies possible data quality issues and suggests what to do before training."
    )

    st.dataframe(
        cleaning_advice,
        use_container_width=True,
    )

    st.subheader("Target Insight")

    if is_valid_target:
        if problem_type == "classification":
            fig_class_balance = create_class_balance_chart(y)
            st.plotly_chart(
                fig_class_balance,
                use_container_width=True,
                key="advisor_class_balance_chart",
            )

            class_counts = y.value_counts()
            min_class = class_counts.min()
            max_class = class_counts.max()
            imbalance_ratio = max_class / max(min_class, 1)

            if imbalance_ratio > 3:
                st.warning(
                    "The target classes appear imbalanced. Accuracy alone may be misleading. "
                    "Weighted F1 Score is more useful in this case."
                )
            else:
                st.success(
                    "The target classes are reasonably balanced for basic model training."
                )

        else:
            fig_target_dist = create_numeric_distribution_chart(
                df.dropna(subset=[target_col]),
                target_col,
            )
            st.plotly_chart(
                fig_target_dist,
                use_container_width=True,
                key="advisor_target_distribution_chart",
            )

            fig_target_corr = create_target_correlation_chart(df, target_col)

            if fig_target_corr is not None:
                st.plotly_chart(
                    fig_target_corr,
                    use_container_width=True,
                    key="advisor_target_correlation_chart",
                )
            else:
                st.info(
                    "Target correlation chart is available only when enough numeric features exist."
                )

    st.subheader("General Guidance")

    st.markdown(
        """
        - Select a target that represents the real output you want to predict.
        - Do not select ID columns, names, serial numbers, or row numbers.
        - For classification, choose a target with categories such as `Yes/No`, `Low/Medium/High`, or `Benign/Malignant`.
        - For regression, choose a numeric target such as a score, price, marks, or risk value.
        - Review missing values and class balance before trusting the model output.
        """
    )

    # ------------------------------------------------------------
    # Tab 2: Advisor continued
    # ------------------------------------------------------------

    selected_target_row = target_recommendations[
        target_recommendations["Column"] == target_col
        ].iloc[0]

    advisor_col_1, advisor_col_2, advisor_col_3 = st.columns(3)

    advisor_col_1.metric("Selected Target", target_col)
    advisor_col_2.metric("Target Status", "Valid" if is_valid_target else "Invalid")
    advisor_col_3.metric(
        "Detected Task",
        problem_type.title() if is_valid_target else "Invalid",
    )

    if is_valid_target:
        st.success(target_message)
    else:
        st.error(target_message)

    recommendation = selected_target_row["Recommendation"]

    if recommendation == "Highly recommended":
        st.success(f"{target_col} is highly recommended as a target column.")
    elif recommendation == "Possible":
        st.info(
            f"{target_col} is possible, but confirm that it is truly the outcome you want to predict."
        )
    elif recommendation == "Weak choice":
        st.warning(
            f"{target_col} is a weak target choice. Prefer a real outcome such as diagnosis, outcome, level, status, class, or score."
        )
    else:
        st.error(f"{target_col} should not be used as a target column.")

    st.subheader("Target Column Recommendations")

    st.dataframe(
        target_recommendations,
        use_container_width=True,
    )

    st.subheader("Data Cleaning Advisor")

    st.write(
        "This table identifies possible data quality issues and suggests what to do before training."
    )

    st.dataframe(
        cleaning_advice,
        use_container_width=True,
    )

    st.subheader("Target Insight")

    if is_valid_target:
        if problem_type == "classification":
            fig_class_balance = create_class_balance_chart(y)
            st.plotly_chart(
                fig_class_balance,
                use_container_width=True,
            )

            class_counts = y.value_counts()
            min_class = class_counts.min()
            max_class = class_counts.max()
            imbalance_ratio = max_class / max(min_class, 1)

            if imbalance_ratio > 3:
                st.warning(
                    "The target classes appear imbalanced. Accuracy alone may be misleading. "
                    "Weighted F1 Score is more useful in this case."
                )
            else:
                st.success(
                    "The target classes are reasonably balanced for basic model training."
                )

        else:
            fig_target_dist = create_numeric_distribution_chart(
                df.dropna(subset=[target_col]),
                target_col,
            )
            st.plotly_chart(
                fig_target_dist,
                use_container_width=True,
            )

            fig_target_corr = create_target_correlation_chart(df, target_col)

            if fig_target_corr is not None:
                st.plotly_chart(
                    fig_target_corr,
                    use_container_width=True,
                )
            else:
                st.info(
                    "Target correlation chart is available only when enough numeric features exist."
                )

    st.subheader("General Guidance")

    st.markdown(
        """
        - Select a target that represents the real output you want to predict.
        - Do not select ID columns, names, serial numbers, or row numbers.
        - For classification, choose a target with categories such as `Yes/No`, `Low/Medium/High`, or `Benign/Malignant`.
        - For regression, choose a numeric target such as a score, price, marks, or risk value.
        - Review missing values and class balance before trusting the model output.
        """
    )

# ------------------------------------------------------------
# Tab 3: Train Models
# ------------------------------------------------------------

with tab_training:
    st.header("Train Models")

    if not is_valid_target:
        st.error("Training is disabled because the selected target column is not suitable.")
        st.stop()

    if len(X) < 25:
        st.error("Dataset is too small for reliable training. Use at least 25 rows.")
        st.stop()

    if y.nunique(dropna=True) < 2:
        st.error("The target has fewer than two unique values. Choose another target.")
        st.stop()

    train_metric_1, train_metric_2, train_metric_3, train_metric_4 = st.columns(4)

    train_metric_1.metric("Target", target_col)
    train_metric_2.metric("Task", problem_type.title())
    train_metric_3.metric("Features Used", X.shape[1])
    train_metric_4.metric("Rows Used", X.shape[0])

    if id_features:
        st.info(
            f"The following ID-like columns were automatically removed from features: {', '.join(id_features)}"
        )

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)

    st.subheader("Preprocessing Pipeline")

    prep_1, prep_2, prep_3, prep_4 = st.columns(4)

    prep_1.success("Missing values handled")
    prep_2.success("Categorical features encoded")
    prep_3.success("Numeric features scaled")
    prep_4.success("Models compared fairly")

    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Candidate Models")

    candidate_models = list(get_candidate_models(problem_type).keys())

    model_card_cols = st.columns(4)

    for index, model_name in enumerate(candidate_models):
        with model_card_cols[index % 4]:
            st.markdown(
                f"""
                    <div class='soft-card'>
                        <div class='mini-title'>{model_name}</div>
                        <div class='mini-text'>Included in automated comparison</div>
                    </div>
                    """,
                unsafe_allow_html=True,
            )

    st.subheader("Training Action")

    st.write(
        "Click the button below to train all candidate models, compare their results, and select the best model."
    )

    run_training = st.button(
        "Run AutoML Training",
        type="primary",
        use_container_width=True,
    )

    if run_training:
        try:
            leaderboard, best_model_name, best_model_info, trained_models = train_models(
                X=X,
                y=y,
                problem_type=problem_type,
                test_size=test_size,
                cv_folds=cv_folds,
            )

            st.session_state["trained_done"] = True
            st.session_state["leaderboard"] = leaderboard
            st.session_state["best_model_name"] = best_model_name
            st.session_state["best_model_info"] = best_model_info
            st.session_state["trained_models"] = trained_models
            st.session_state["problem_type"] = problem_type
            st.session_state["target_col"] = target_col
            st.session_state["X"] = X
            st.session_state["y"] = y
            st.session_state["df"] = df
            st.session_state["quality_score"] = quality_score
            st.session_state["quality_label"] = quality_label
            st.session_state["id_features"] = id_features
            st.session_state["cleaning_advice"] = cleaning_advice

            st.success("Training completed. Open the Results tab to view the dashboard.")

        except Exception as error:
            st.error(f"Training failed: {error}")

# ------------------------------------------------------------
# Tab 4: Results
# ------------------------------------------------------------

with tab_results:
    st.header("Results Dashboard")

    if not st.session_state.get("trained_done", False):
        st.warning("Run model training first from the Train Models tab.")

    else:
        leaderboard = st.session_state["leaderboard"]
        best_model_name = st.session_state["best_model_name"]
        best_model_info = st.session_state["best_model_info"]
        problem_type_state = st.session_state["problem_type"]
        target_state = st.session_state["target_col"]

        successful_models = leaderboard[leaderboard["Status"] == "Success"].copy()
        best_score = float(successful_models.iloc[0]["Main Score"])

        st.markdown("<div class='winner-card'>", unsafe_allow_html=True)

        st.subheader(f"Best Model: {best_model_name}")
        st.write(model_explanation(best_model_name, problem_type_state))

        result_metric_1, result_metric_2, result_metric_3 = st.columns(3)

        result_metric_1.metric("Selected Target", target_state)
        result_metric_2.metric("Task", problem_type_state.title())
        result_metric_3.metric("Best Score", round(best_score, 4))

        st.markdown("</div>", unsafe_allow_html=True)

        st.subheader("Model Leaderboard")

        display_leaderboard = leaderboard.copy()

        numeric_cols = display_leaderboard.select_dtypes(include=[np.number]).columns
        display_leaderboard[numeric_cols] = display_leaderboard[numeric_cols].round(4)

        st.dataframe(
            display_leaderboard,
            use_container_width=True,
        )

        st.subheader("Model Performance Comparison")

        plot_leaderboard = successful_models.sort_values(
            "Main Score",
            ascending=True,
        )

        fig_leaderboard = px.bar(
            plot_leaderboard,
            x="Main Score",
            y="Model",
            orientation="h",
            text="Main Score",
            title="Model Leaderboard by Main Score",
        )

        fig_leaderboard.update_traces(
            texttemplate="%{text:.3f}",
            textposition="outside",
        )

        fig_leaderboard.update_layout(
            height=430,
            margin=dict(l=10, r=30, t=50, b=20),
        )

        st.plotly_chart(
            fig_leaderboard,
            use_container_width=True,
            key="results_model_leaderboard_chart",
        )

        st.subheader("Model Insight")

        st.info(
            generate_model_insight(
                leaderboard=leaderboard,
                best_model_name=best_model_name,
                problem_type=problem_type_state,
            )
        )

        visual_left, visual_right = st.columns([1, 1])

        with visual_left:
            st.subheader("Evaluation Visual")

            y_test = best_model_info["y_test"]
            y_pred = best_model_info["y_pred"]

            if problem_type_state == "classification":
                fig_cm, ax = plt.subplots(figsize=(6, 4.5))

                cm = confusion_matrix(y_test, y_pred)
                display = ConfusionMatrixDisplay(confusion_matrix=cm)

                display.plot(
                    ax=ax,
                    values_format="d",
                )

                ax.set_title("Confusion Matrix")

                st.pyplot(fig_cm)

            else:
                evaluation_df = pd.DataFrame(
                    {
                        "Actual": y_test,
                        "Predicted": y_pred,
                    }
                )

                fig_scatter = px.scatter(
                    evaluation_df,
                    x="Actual",
                    y="Predicted",
                    title="Actual vs Predicted Values",
                )

                min_value = min(
                    evaluation_df["Actual"].min(),
                    evaluation_df["Predicted"].min(),
                )

                max_value = max(
                    evaluation_df["Actual"].max(),
                    evaluation_df["Predicted"].max(),
                )

                fig_scatter.add_shape(
                    type="line",
                    x0=min_value,
                    y0=min_value,
                    x1=max_value,
                    y1=max_value,
                    line=dict(dash="dash"),
                )

                fig_scatter.update_layout(
                    height=420,
                )

                st.plotly_chart(
                    fig_scatter,
                    use_container_width=True,
                    key="results_actual_vs_predicted_chart",
                )
        with visual_right:
            st.subheader("Explainability")

            feature_importance = create_feature_importance(best_model_info)

            if feature_importance is not None and not feature_importance.empty:
                fig_importance = px.bar(
                    feature_importance.sort_values("Importance", ascending=True),
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title="Top Feature Importance",
                )

                fig_importance.update_layout(
                    height=420,
                    margin=dict(l=10, r=20, t=50, b=20),
                )

                st.plotly_chart(
                    fig_importance,
                    use_container_width=True,
                    key="results_feature_importance_chart",
                )

            else:
                st.info(
                    "Feature importance is not directly available for this selected model. "
                    "Tree-based models such as Random Forest and Gradient Boosting usually provide feature importance."
                )

        st.subheader("Plain-English Interpretation")

        if problem_type_state == "classification":
            st.success(
                f"The system selected {best_model_name} because it achieved the strongest "
                f"Weighted F1 Score among the tested classifiers. This means it performed "
                f"best overall across the target classes."
            )

        else:
            st.success(
                f"The system selected {best_model_name} because it achieved the strongest "
                f"R² Score among the tested regressors. This means it explained the target "
                f"value better than the other tested models."
            )

# ------------------------------------------------------------
# Tab 5: Predict
# ------------------------------------------------------------

with tab_predict:
    st.header("Predict New Case")

    if not st.session_state.get("trained_done", False):
        st.warning("Train the models first before using prediction.")

    else:
        X_state = st.session_state["X"]
        best_model_info = st.session_state["best_model_info"]
        target_state = st.session_state["target_col"]

        st.write(
            f"Enter values below. The app will use the best selected model to predict **{target_state}**."
        )

        input_data = {}

        with st.form("prediction_form"):
            for col in X_state.columns:
                if pd.api.types.is_numeric_dtype(X_state[col]):
                    median_value = X_state[col].median()

                    if pd.isna(median_value):
                        median_value = 0.0

                    input_data[col] = st.number_input(
                        col,
                        value=float(median_value),
                    )

                else:
                    unique_values = (
                        X_state[col]
                        .dropna()
                        .astype(str)
                        .unique()
                        .tolist()
                    )

                    if 1 <= len(unique_values) <= 30:
                        input_data[col] = st.selectbox(
                            col,
                            unique_values,
                        )

                    else:
                        default_text = unique_values[0] if unique_values else "Unknown"

                        input_data[col] = st.text_input(
                            col,
                            value=default_text,
                        )

            submit_prediction = st.form_submit_button(
                "Predict",
                type="primary",
            )

        if submit_prediction:
            input_df = pd.DataFrame([input_data])
            prediction = best_model_info["pipeline"].predict(input_df)[0]

            st.markdown("<div class='winner-card'>", unsafe_allow_html=True)

            st.subheader("Prediction Result")
            st.metric("Predicted Output", str(prediction))

            st.markdown("</div>", unsafe_allow_html=True)

            st.subheader("Input Record Used")
            st.dataframe(
                input_df,
                use_container_width=True,
            )

# ------------------------------------------------------------
# Tab 6: Export
# ------------------------------------------------------------

with tab_export:
    st.header("Export Report and Files")

    if not st.session_state.get("trained_done", False):
        st.warning("Train the models first to generate export files.")

    else:
        leaderboard = st.session_state["leaderboard"]
        best_model_name = st.session_state["best_model_name"]
        best_model_info = st.session_state["best_model_info"]
        problem_type_state = st.session_state["problem_type"]
        target_state = st.session_state["target_col"]
        quality_score_state = st.session_state["quality_score"]
        quality_label_state = st.session_state["quality_label"]
        id_features_state = st.session_state["id_features"]
        cleaning_advice_state = st.session_state["cleaning_advice"]

        successful_models = leaderboard[leaderboard["Status"] == "Success"]
        best_score = float(successful_models.iloc[0]["Main Score"])

        report = create_report(
            df=df,
            target_col=target_state,
            problem_type=problem_type_state,
            leaderboard=leaderboard,
            best_model_name=best_model_name,
            best_score=best_score,
            quality_score=quality_score_state,
            quality_label=quality_label_state,
            id_features=id_features_state,
            cleaning_advice=cleaning_advice_state,
        )

        st.subheader("Project Report Preview")

        st.text_area(
            "Report",
            report,
            height=430,
        )

        export_col_1, export_col_2, export_col_3, export_col_4 = st.columns(4)

        with export_col_1:
            st.download_button(
                "Download Report (.txt)",
                data=report,
                file_name="smart_automl_pro_report.txt",
                mime="text/plain",
                use_container_width=True,
            )

        with export_col_2:
            st.download_button(
                "Download Leaderboard (.csv)",
                data=dataframe_download(leaderboard),
                file_name="model_leaderboard.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with export_col_3:
            st.download_button(
                "Download Cleaning Advice (.csv)",
                data=dataframe_download(cleaning_advice_state),
                file_name="data_cleaning_advice.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with export_col_4:
            st.download_button(
                "Download Best Model (.pkl)",
                data=download_model_bytes(best_model_info["pipeline"]),
                file_name=f"best_model_{best_model_name.replace(' ', '_').lower()}.pkl",
                mime="application/octet-stream",
                use_container_width=True,
            )

        st.subheader("Presentation Line")

        st.success(
            "This project is not a single prediction model. It is a visual AutoML system "
            "that recommends a target, detects the ML task, analyzes data quality, preprocesses "
            "the dataset, trains multiple algorithms, compares performance, explains the best model, "
            "supports prediction, and exports results."
        )