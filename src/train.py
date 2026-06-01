import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.data_processing import build_feature_pipeline, RFMProxyTargetGenerator

def run_model_training_workflow(data_path):
    """
    Task 5: Complete model training, tracking, tuning, and registration workflow.
    """
    # Start MLflow experiment tracking
    mlflow.set_experiment("Bati_Bank_Credit_Risk_Project")
    
    # 1. Load Raw Dataset Workspace
    print(">>> Loading transaction logs data...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(">>> Target CSV path missing. Synthesizing structural baseline mock dataframe...")
        np.random.seed(42)
        records = 2000
        df = pd.DataFrame({
            'TransactionId': [f'T{i}' for i in range(records)],
            'CustomerId': [f'C{np.random.randint(1, 200)}' for i in range(records)],
            'Amount': np.random.exponential(scale=3000, size=records) * np.random.choice([1, -0.1], records),
            'ProductCategory': np.random.choice(['airtime', 'utility', 'financial_services'], records),
            'ChannelId': np.random.choice(['Channel_1', 'Channel_3'], records),
            'PricingStrategy': np.random.choice([1, 2, 4], records),
            'TransactionStartTime': pd.date_range(start='2026-01-01', periods=records, freq='T').strftime('%Y-%m-%d %H:%M:%S')
        })
        df['Value'] = df['Amount'].abs()

    # 2. Extract Proxy Target Labels (Task 4)
    print(">>> Engineering RFM proxy target classifications...")
    target_generator = RFMProxyTargetGenerator(n_clusters=3, random_state=42)
    labeled_df = target_generator.fit_transform(df)
    
    # Isolate targets and features
    X = labeled_df.drop(columns=['is_high_risk'])
    y = labeled_df['is_high_risk']
    
    # 3. Stratified Train-Test Splitting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # 4. Fit Feature Transformation Pipelines (Task 3)
    feature_pipeline = build_feature_pipeline()
    X_train_transformed = feature_pipeline.fit_transform(X_train)
    X_test_transformed = feature_pipeline.transform(X_test)
    
    # 5. Define Model Evaluation Grid Parameters
    models_to_evaluate = {
        "Logistic_Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "params": {"C": [0.1, 1.0, 10.0]}
        },
        "Random_Forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {"n_estimators": [50, 100], "max_depth": [5, 10]}
        },
        "XGBoost": {
            "model": XGBClassifier(eval_metric='logloss', random_state=42),
            "params": {"max_depth": [3, 6], "learning_rate": [0.05, 0.2]}
        }
    }
    
    best_overall_score = -1
    best_model_name = None
    
    # 6. Execute Run Optimizations via Hyperparameter Grid Searches
    for name, config in models_to_evaluate.items():
        with mlflow.start_run(run_name=name):
            print(f">>> Optimizing hyperparameters for: {name}")
            grid_search = GridSearchCV(config["model"], config["params"], cv=3, scoring='roc_auc', n_jobs=-1)
            grid_search.fit(X_train_transformed, y_train)
            
            best_model = grid_search.best_estimator__
            predictions = best_model.predict(X_test_transformed)
            probabilities = best_model.predict_proba(X_test_transformed)[:, 1]
            
            # Metrics Evaluation
            acc = accuracy_score(y_test, predictions)
            prec = precision_score(y_test, predictions, zero_division=0)
            rec = recall_score(y_test, predictions, zero_division=0)
            f1 = f1_score(y_test, predictions, zero_division=0)
            auc = roc_auc_score(y_test, probabilities)
            
            # Log Parameters and Metrics directly to MLflow Tracking server
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metric("Accuracy", acc)
            mlflow.log_metric("Precision", prec)
            mlflow.log_metric("Recall", rec)
            mlflow.log_metric("F1_Score", f1)
            mlflow.log_metric("ROC_AUC", auc)
            
            # Log Model Artifact Binaries
            mlflow.sklearn.log_model(best_model, artifact_path=f"models/{name}")
            print(f"Results [{name}] -> ROC-AUC: {auc:.4f} | F1-Score: {f1:.4f}")
            
            # Track the top-performing model
            if auc > best_overall_score:
                best_overall_score = auc
                best_model_name = name
                champion_model_object = best_model
                
    # 7. Register Champion Production Model to Registry
    print(f"\n>>> Registration Step: Model [{best_model_name}] selected as Champion runtime.")
    with mlflow.start_run(run_name="Champion_Registration") as run:
        mlflow.log_param("Champion_Architecture", best_model_name)
        mlflow.log_metric("Max_ROC_AUC", best_overall_score)
        # Register the model artifact
        mlflow.sklearn.log_model(
            sk_model=champion_model_object,
            artifact_path="champion_model",
            registered_model_name="champion_credit_model"
        )
    print(">>> Model tracking and registration completed.")

if __name__ == "__main__":
    run_model_training_workflow(data_path="data/raw/Xente_Variable_Data.csv")