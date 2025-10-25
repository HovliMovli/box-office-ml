End-to-end TMDB box office prediction using Random Forest and XGBoost. Run main_ml_file.py in the same folder as train.csv (80/20 split, seed=42) to train/evaluate, then save figures (correlation, actual vs predicted, residuals, feature importances, ROC/PR, confusion matrix) and CSVs (comparison_results_regression.csv, classification_metrics.csv, submission.csv). Reports regression metrics (R², MSE, RMSE, MAE, PCC) and optional classification metrics (Accuracy, Precision, Recall, F1, AUROC, PR-AUC, MCC).


How to run:

# from your project folder
python3 -m venv .venv
source .venv/bin/activate

# upgrade installer tools
python -m pip install --upgrade pip setuptools wheel

# minimal deps (your script will fall back to HGB if boosters aren’t present)
pip install numpy pandas scikit-learn

# optional boosters (install if you want faster/better models)
pip install xgboost lightgbm catboost seaborn

# run your script
python run_model.py --train_path train.csv --test_path test.csv --out_path submission.csv

# when done
deactivate
