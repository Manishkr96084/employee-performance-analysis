"""
Employee Performance & Satisfaction Analysis - Main Execution Pipeline
Author: Swayam / Final Year Project Team
Certifier Alignment: IABAC Certified Data Scientist Project

This script runs the entire machine learning and analytics workflow:
1. Data Preprocessing & Cleaning (raw Excel -> cleaned CSV)
2. Model Training & Evaluation (Random Forest Classifier -> rf_model.pkl)
3. Model Batch Prediction (predictions.csv)
4. Plot & Figure Generation (saved to reports/figures/)
5. Optional: Notebook validation and execution (--notebooks)
6. Optional: Launch interactive Streamlit web dashboard (--app)
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for generating figures cleanly
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Ensure UTF-8 output on Windows terminal
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'employee_data.xlsx')
CLEANED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'employee_data_cleaned.csv')
PREDICTIONS_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'predictions.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'src', 'models', 'rf_model.pkl')
FIGURES_DIR = os.path.join(BASE_DIR, 'reports', 'figures')

CAT_COLS = [
    'Gender', 'EducationBackground', 'MaritalStatus',
    'EmpDepartment', 'EmpJobRole', 'BusinessTravelFrequency',
    'OverTime', 'Attrition'
]

def step_process_data():
    print("\n" + "=" * 60)
    print(" [STEP 1/4] DATA PREPROCESSING & CLEANING")
    print("=" * 60)

    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Raw data file not found at: {RAW_DATA_PATH}")

    print(f"--> Loading raw dataset from: {RAW_DATA_PATH}")
    df = pd.read_excel(RAW_DATA_PATH)
    print(f"--> Raw dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # Missing value handling
    missing_count = df.isnull().sum().sum()
    print(f"--> Total missing values detected: {missing_count}")
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('Unknown')
        else:
            df[col] = df[col].fillna(0)

    # Encode categorical variables
    print(f"--> Applying LabelEncoder to categorical features: {', '.join(CAT_COLS)}")
    le = LabelEncoder()
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    os.makedirs(os.path.dirname(CLEANED_DATA_PATH), exist_ok=True)
    df.to_csv(CLEANED_DATA_PATH, index=False)
    print(f"--> Processed dataset successfully saved to: {CLEANED_DATA_PATH}")
    return df

def step_generate_visualizations(df=None):
    print("\n" + "=" * 60)
    print(" [STEP 2/4] EXPLORATORY ANALYSIS & VISUALIZATION")
    print("=" * 60)

    if df is None:
        if not os.path.exists(CLEANED_DATA_PATH):
            df = step_process_data()
        else:
            df = pd.read_csv(CLEANED_DATA_PATH)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    sns.set_theme(style='whitegrid')

    # 1. Job Satisfaction Distribution
    plt.figure(figsize=(7, 4.5))
    ax = sns.countplot(x='EmpJobSatisfaction', data=df, palette='viridis')
    plt.title('Distribution of Employee Job Satisfaction (1: Low to 4: High)', fontsize=12, pad=12)
    plt.xlabel('Job Satisfaction Level')
    plt.ylabel('Employee Count')
    plt.tight_layout()
    p1 = os.path.join(FIGURES_DIR, '01_satisfaction_distribution.png')
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"--> Saved: {p1}")

    # 2. Satisfaction by Department
    plt.figure(figsize=(8, 5))
    sns.barplot(x='EmpDepartment', y='EmpJobSatisfaction', data=df, estimator=np.mean, palette='coolwarm')
    plt.title('Average Job Satisfaction by Department (Encoded)', fontsize=12, pad=12)
    plt.ylabel('Average Satisfaction (1-4)')
    plt.xlabel('Department Code')
    plt.tight_layout()
    p2 = os.path.join(FIGURES_DIR, '02_satisfaction_by_department.png')
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"--> Saved: {p2}")

    # 3. OverTime vs Satisfaction
    plt.figure(figsize=(7, 4.5))
    sns.violinplot(x='OverTime', y='EmpJobSatisfaction', data=df, palette='Blues')
    plt.title('Job Satisfaction Distribution by OverTime Status (0=No, 1=Yes)', fontsize=12, pad=12)
    plt.xlabel('OverTime Status')
    plt.ylabel('Job Satisfaction')
    plt.tight_layout()
    p3 = os.path.join(FIGURES_DIR, '03_satisfaction_by_overtime.png')
    plt.savefig(p3, dpi=300)
    plt.close()
    print(f"--> Saved: {p3}")

    # 4. Correlation Heatmap
    plt.figure(figsize=(14, 10))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=False, cmap='coolwarm', square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Heatmap', fontsize=14, pad=12)
    plt.tight_layout()
    p4 = os.path.join(FIGURES_DIR, '04_correlation_heatmap.png')
    plt.savefig(p4, dpi=300)
    plt.close()
    print(f"--> Saved: {p4}")

    # 5. Performance Rating by Department
    if 'PerformanceRating' in df.columns:
        plt.figure(figsize=(8, 5))
        sns.boxplot(x='EmpDepartment', y='PerformanceRating', data=df, palette='mako')
        plt.title('Performance Rating by Department Code', fontsize=12, pad=12)
        plt.tight_layout()
        p5 = os.path.join(FIGURES_DIR, '05_performance_by_department.png')
        plt.savefig(p5, dpi=300)
        plt.close()
        print(f"--> Saved: {p5}")

def step_train_model(df=None):
    print("\n" + "=" * 60)
    print(" [STEP 3/4] MODEL TRAINING & EVALUATION (Random Forest)")
    print("=" * 60)

    if df is None:
        if not os.path.exists(CLEANED_DATA_PATH):
            df = step_process_data()
        else:
            df = pd.read_csv(CLEANED_DATA_PATH)

    # Define binary target: LowSatisfaction = 1 if JobSatisfaction <= 2, else 0
    df_model = df.copy()
    df_model['LowSatisfaction'] = (df_model['EmpJobSatisfaction'] <= 2).astype(int)

    target = 'LowSatisfaction'
    X = df_model.drop(columns=['EmpNumber', 'EmpJobSatisfaction', target], errors='ignore')
    y = df_model[target]

    print(f"--> Target Variable: '{target}' (1=Low Satisfaction, 0=High/Normal Satisfaction)")
    print(f"--> Target Class Breakdown:\n{y.value_counts().to_string()}")
    print(f"--> Feature Count: {X.shape[1]} features")

    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"--> Train set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("\n--- MODEL PERFORMANCE RESULTS ---")
    print(f"Model Accuracy: {acc * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    # Feature Importance
    importances = rf.feature_importances_
    feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=True)

    print("Top 5 Driving Features for Predicting Low Satisfaction:")
    for feat, val in feat_imp.tail(5)[::-1].items():
        print(f"  • {feat:28s}: {val:.4f}")

    # Plot & Save Feature Importance
    plt.figure(figsize=(10, 6.5))
    feat_imp.plot(kind='barh', color='teal')
    plt.title('Feature Importance for Predicting Low Satisfaction (Random Forest)', fontsize=12, pad=12)
    plt.xlabel('Importance Score')
    plt.tight_layout()
    p_imp = os.path.join(FIGURES_DIR, '06_feature_importance.png')
    plt.savefig(p_imp, dpi=300)
    plt.close()
    print(f"--> Feature importance plot saved to: {p_imp}")

    # Save trained model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(rf, MODEL_PATH)
    print(f"--> Trained model serialized and saved to: {MODEL_PATH}")

    return rf, X

def step_predict_model(rf=None, df=None):
    print("\n" + "=" * 60)
    print(" [STEP 4/4] BATCH PREDICTION PIPELINE")
    print("=" * 60)

    if rf is None:
        if not os.path.exists(MODEL_PATH):
            rf, _ = step_train_model(df)
        else:
            print(f"--> Loading existing model from: {MODEL_PATH}")
            rf = joblib.load(MODEL_PATH)

    if df is None:
        df = pd.read_csv(CLEANED_DATA_PATH)

    new_data = df.copy()
    X_new = new_data.drop(columns=['EmpNumber', 'EmpJobSatisfaction', 'LowSatisfaction', 'PredictedLowSatisfaction'], errors='ignore')

    predictions = rf.predict(X_new)
    probabilities = rf.predict_proba(X_new)[:, 1] if hasattr(rf, 'predict_proba') else None

    new_data['PredictedLowSatisfaction'] = predictions
    if probabilities is not None:
        new_data['RiskProbability'] = probabilities
        new_data['HighRiskFlag'] = (probabilities > 0.60).astype(int)

    new_data.to_csv(PREDICTIONS_PATH, index=False)
    print(f"--> Predictions saved to: {PREDICTIONS_PATH}")
    print(f"--> Total employees evaluated: {len(new_data)}")
    print(f"--> Flagged as Low Satisfaction Risk: {(predictions == 1).sum()} employees")
    if probabilities is not None:
        print(f"--> Flagged as High Risk (probability > 60%): {(probabilities > 0.60).sum()} employees")

    print("\nSample Preview of Predictions:")
    display_cols = ['EmpNumber', 'PredictedLowSatisfaction']
    if 'RiskProbability' in new_data.columns:
        display_cols.extend(['RiskProbability', 'HighRiskFlag'])
    print(new_data[display_cols].head(10).to_string(index=False))

def step_execute_notebooks():
    print("\n" + "=" * 60)
    print(" [OPTIONAL] VALIDATING & EXECUTING JUPYTER NOTEBOOKS")
    print("=" * 60)
    import nbformat
    from nbclient import NotebookClient

    notebooks = [
        'src/Data_Processing/data_processing.ipynb',
        'src/Data_Processing/data_exploratory_analysis.ipynb',
        'src/models/train_model.ipynb',
        'src/models/predict_model.ipynb',
        'src/visualization/visualize.ipynb'
    ]

    for rel_path in notebooks:
        full_path = os.path.join(BASE_DIR, rel_path)
        dir_name = os.path.dirname(full_path)
        print(f"--> Executing: {rel_path} ...", end=" ", flush=True)
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
            client = NotebookClient(nb, timeout=600, kernel_name='python3', resources={'metadata': {'path': dir_name}})
            client.execute()
            with open(full_path, 'w', encoding='utf-8') as f:
                nbformat.write(nb, f)
            print("OK [Validated & Saved]")
        except Exception as e:
            print(f"FAILED: {e}")

def main():
    parser = argparse.ArgumentParser(description="Employee Performance Analysis - Complete Execution Pipeline")
    parser.add_argument(
        '--step',
        choices=['all', 'process', 'visualize', 'train', 'predict'],
        default='all',
        help="Specific pipeline step to execute (default: all)"
    )
    parser.add_argument(
        '--notebooks',
        action='store_true',
        help="Re-execute and update all Jupyter notebooks in place"
    )
    parser.add_argument(
        '--app',
        action='store_true',
        help="Launch the interactive Streamlit web dashboard after execution"
    )

    args = parser.parse_args()

    print("\n" + "*" * 65)
    print("   EMPLOYEE PERFORMANCE & SATISFACTION ANALYSIS PIPELINE")
    print("   IABAC Certified Data Scientist Project Reference")
    print("*" * 65)

    if args.step in ['all', 'process']:
        df = step_process_data()
    else:
        df = pd.read_csv(CLEANED_DATA_PATH) if os.path.exists(CLEANED_DATA_PATH) else None

    if args.step in ['all', 'visualize']:
        step_generate_visualizations(df)

    if args.step in ['all', 'train']:
        rf, _ = step_train_model(df)
    else:
        rf = None

    if args.step in ['all', 'predict']:
        step_predict_model(rf, df)

    if args.notebooks:
        step_execute_notebooks()

    print("\n" + "=" * 60)
    print(" [COMPLETED] Entire pipeline executed successfully!")
    print(f" - Cleaned Data:   {CLEANED_DATA_PATH}")
    print(f" - Trained Model:  {MODEL_PATH}")
    print(f" - Predictions:    {PREDICTIONS_PATH}")
    print(f" - Figures/Plots:  {FIGURES_DIR}")
    print("=" * 60)

    if args.app:
        print("\n--> Launching Streamlit web application...")
        os.system("streamlit run app.py")

if __name__ == '__main__':
    main()
