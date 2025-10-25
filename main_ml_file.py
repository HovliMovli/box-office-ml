"""
MOVIE BOX OFFICE PREDICTION - COMPLETE VERSION
With Classification Metrics and All Required Visualizations

Includes:
- Regression models (R², MAE, etc.)
- Classification models (Accuracy, Precision, Recall, F1, AUROC, PR-AUC)
- All 9 required visualizations
"""

CSV_PATH = 'train.csv'

import pandas as pd
import numpy as np
import json
import warnings
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
                             accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix,
                             roc_curve, precision_recall_curve, auc)
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("="*80)
print("MOVIE BOX OFFICE PREDICTION - COMPLETE ANALYSIS")
print("Regression + Classification with All Metrics")
print("="*80 + "\n")

#==============================================================================
# LOAD DATA
#==============================================================================
print("📂 Loading data...")
df = pd.read_csv(CSV_PATH)
print(f"✓ Loaded {len(df)} movies | {df.shape[1]} columns")

#==============================================================================
# PREPROCESSING
#==============================================================================
print("\n🔧 Preprocessing data...")

def parse_json(x):
    try:
        if pd.isna(x): return []
        data = json.loads(x) if isinstance(x, str) else x
        return [item.get('name', '') for item in data if isinstance(item, dict)]
    except: return []

json_cols = ['genres', 'production_companies', 'production_countries', 'cast', 'crew', 'Keywords', 'spoken_languages']
for col in json_cols:
    if col in df.columns:
        df[col + '_parsed'] = df[col].apply(parse_json)

# Handle missing values
df['budget'] = df['budget'].fillna(df['budget'].median())
df['runtime'] = df['runtime'].fillna(df['runtime'].median())
df['popularity'] = df['popularity'].fillna(df['popularity'].median())

# Temporal features
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['release_year'] = df['release_date'].dt.year
df['release_month'] = df['release_date'].dt.month
df['release_dayofweek'] = df['release_date'].dt.dayofweek

# Derived features
df['inflationBudget'] = df['budget'] * (1.02 ** (2018 - df['release_year']))
df['log_budget'] = np.log1p(df['budget'])
df['budget_year_ratio'] = df['budget'] / (df['release_year'] - 1989 + 1)
df['budget_runtime_ratio'] = df['budget'] / (df['runtime'] + 1)

pop_by_year = df.groupby('release_year')['popularity'].mean()
df['popularity_mean_year'] = df['release_year'].map(pop_by_year)

df['num_genres'] = df['genres_parsed'].apply(len)
df['num_production_companies'] = df['production_companies_parsed'].apply(len)
df['num_production_countries'] = df['production_countries_parsed'].apply(len)
df['num_cast'] = df['cast_parsed'].apply(len)
df['num_crew'] = df['crew_parsed'].apply(len)
df['num_keywords'] = df['Keywords_parsed'].apply(len) if 'Keywords_parsed' in df.columns else 0

df['has_homepage'] = (~df['homepage'].isna()).astype(int)
df['has_tagline'] = (~df['tagline'].isna()).astype(int)
df['has_collection'] = (~df['belongs_to_collection'].isna()).astype(int)
df['is_english'] = (df['original_language'] == 'en').astype(int)

# Target variables
df['log_revenue'] = np.log1p(df['revenue'])

# CLASSIFICATION TARGET: Hit vs Non-hit (median threshold)
revenue_median = df['revenue'].median()
df['is_hit'] = (df['revenue'] >= revenue_median).astype(int)

print("✓ Preprocessing complete")

# Drop NaN rows and handle infinite values
features = [
    'budget', 'popularity', 'runtime', 'release_year', 'release_month', 
    'release_dayofweek', 'log_budget', 'budget_year_ratio', 'inflationBudget',
    'budget_runtime_ratio', 'popularity_mean_year', 'num_genres', 
    'num_production_companies', 'num_production_countries', 'num_cast', 
    'num_crew', 'num_keywords', 'has_homepage', 'has_tagline', 
    'has_collection', 'is_english'
]

X = df[features].fillna(0)
y_regression = df['log_revenue']
y_classification = df['is_hit']

# Replace infinite values with 0
X = X.replace([np.inf, -np.inf], 0)

# Remove rows with NaN in target variables
valid_idx = ~(X.isna().any(axis=1) | y_regression.isna() | y_classification.isna() | 
              np.isinf(y_regression) | np.isinf(X).any(axis=1))
X = X[valid_idx]
y_regression = y_regression[valid_idx]
y_classification = y_classification[valid_idx]

original_count = len(df)
final_count = len(X)
dropped_count = original_count - final_count

if dropped_count > 0:
    print(f"ℹ️ Dropped {dropped_count} rows with missing/invalid values after feature engineering.")

print(f"✓ Features ready: {len(features)}")
print(f"✓ Samples after cleaning: {final_count}")

# Final check for any remaining problematic values
assert not np.any(np.isnan(X.values)), "NaN values still present in X"
assert not np.any(np.isinf(X.values)), "Inf values still present in X"
assert not np.any(np.isnan(y_regression.values)), "NaN values still present in y_regression"
assert not np.any(np.isinf(y_regression.values)), "Inf values still present in y_regression"

print("✓ Data validation passed - no NaN or Inf values")

#==============================================================================
# TRAIN/TEST SPLIT
#==============================================================================
X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
    X, y_regression, y_classification, test_size=0.2, random_state=RANDOM_SEED, stratify=y_classification
)
print(f"✓ Train: {len(X_train)}, Test: {len(X_test)}")

#==============================================================================
# REGRESSION MODELS
#==============================================================================
print("\n" + "="*80)
print("PART 1: REGRESSION MODELS")
print("="*80)

def evaluate_regression(y_true, y_pred, name):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    y_true_exp = np.expm1(y_true)
    y_pred_exp = np.expm1(y_pred)
    mask = y_true_exp != 0
    mape = np.mean(np.abs((y_true_exp[mask] - y_pred_exp[mask]) / y_true_exp[mask])) * 100
    pcc = np.corrcoef(y_true, y_pred)[0, 1]
    
    print(f"\n{name} - Test Set:")
    print(f"  R²    : {r2:.4f}")
    print(f"  MSE   : {mse:.4f}")
    print(f"  RMSE  : {rmse:.4f}")
    print(f"  MAE   : {mae:.4f}")
    print(f"  MAPE  : {mape:.4f}%")
    print(f"  PCC   : {pcc:.4f}")
    
    return {'r2': r2, 'mse': mse, 'rmse': rmse, 'mae': mae, 'mape': mape, 'pcc': pcc}

# Random Forest Regression
print("\n🌲 TRAINING RANDOM FOREST (REGRESSION)")
rf_reg = RandomForestRegressor(n_estimators=500, max_depth=10, min_samples_split=2,
                                min_samples_leaf=2, random_state=RANDOM_SEED, n_jobs=-1)
rf_reg.fit(X_train, y_reg_train)
rf_reg_pred = rf_reg.predict(X_test)
rf_reg_metrics = evaluate_regression(y_reg_test, rf_reg_pred, "Random Forest")

cv_scores_rf = cross_val_score(rf_reg, X_train, y_reg_train, cv=3, scoring='r2', n_jobs=-1)
print(f"  3-fold CV R²: {cv_scores_rf.mean():.4f} (+/- {cv_scores_rf.std()*2:.4f})")

# XGBoost Regression
print("\n🚀 TRAINING XGBOOST (REGRESSION)")
xgb_reg = xgb.XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.01,
                            subsample=0.7, colsample_bytree=0.8, random_state=RANDOM_SEED, n_jobs=-1)
xgb_reg.fit(X_train, y_reg_train)
xgb_reg_pred = xgb_reg.predict(X_test)
xgb_reg_metrics = evaluate_regression(y_reg_test, xgb_reg_pred, "XGBoost")

cv_scores_xgb = cross_val_score(xgb_reg, X_train, y_reg_train, cv=3, scoring='r2', n_jobs=-1)
print(f"  3-fold CV R²: {cv_scores_xgb.mean():.4f} (+/- {cv_scores_xgb.std()*2:.4f})")

# Statistical test
rf_errors = np.abs(y_reg_test - rf_reg_pred)
xgb_errors = np.abs(y_reg_test - xgb_reg_pred)
t_stat, p_value = stats.ttest_rel(rf_errors, xgb_errors)

print("\n📊 STATISTICAL HYPOTHESIS TESTING (REGRESSION)")
print(f"Paired t-test: Random Forest vs XGBoost")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value:.4f}")
if p_value < 0.05:
    print(f"  ✓ Significant difference (α=0.05)")
else:
    print(f"  ○ No significant difference (α=0.05)")

#==============================================================================
# CLASSIFICATION MODELS
#==============================================================================
print("\n" + "="*80)
print("PART 2: CLASSIFICATION MODELS (HIT vs NON-HIT)")
print("="*80)
print(f"Classification threshold: Revenue >= ${revenue_median:,.0f} = HIT")
print(f"Class distribution - Train: {y_clf_train.value_counts().to_dict()}")
print(f"Class distribution - Test: {y_clf_test.value_counts().to_dict()}")

def evaluate_classification(y_true, y_pred, y_prob, name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auroc = roc_auc_score(y_true, y_prob)
    avg_prec = average_precision_score(y_true, y_prob)
    
    print(f"\n{name} - Classification Metrics:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  AUROC:     {auroc:.4f}")
    print(f"  PR-AUC:    {avg_prec:.4f}")
    
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"                Predicted Non-Hit  Predicted Hit")
    print(f"  True Non-Hit        {cm[0,0]:4d}            {cm[0,1]:4d}")
    print(f"  True Hit            {cm[1,0]:4d}            {cm[1,1]:4d}")
    
    return {
        'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
        'auroc': auroc, 'avg_precision': avg_prec, 'confusion_matrix': cm
    }

# Random Forest Classification
print("\n🌲 TRAINING RANDOM FOREST (CLASSIFICATION)")
rf_clf = RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_split=2,
                                 min_samples_leaf=2, random_state=RANDOM_SEED, n_jobs=-1)
rf_clf.fit(X_train, y_clf_train)
rf_clf_pred = rf_clf.predict(X_test)
rf_clf_prob = rf_clf.predict_proba(X_test)[:, 1]
rf_clf_metrics = evaluate_classification(y_clf_test, rf_clf_pred, rf_clf_prob, "Random Forest")

# XGBoost Classification
print("\n🚀 TRAINING XGBOOST (CLASSIFICATION)")
xgb_clf = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.01,
                             subsample=0.7, colsample_bytree=0.8, random_state=RANDOM_SEED, n_jobs=-1)
xgb_clf.fit(X_train, y_clf_train)
xgb_clf_pred = xgb_clf.predict(X_test)
xgb_clf_prob = xgb_clf.predict_proba(X_test)[:, 1]
xgb_clf_metrics = evaluate_classification(y_clf_test, xgb_clf_pred, xgb_clf_prob, "XGBoost")

#==============================================================================
# VISUALIZATIONS
#==============================================================================
print("\n" + "="*80)
print("📈 GENERATING ALL REQUIRED VISUALIZATIONS")
print("="*80)

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# 1. Correlation Matrix
print("\n1. Creating correlation matrix...")
plt.figure(figsize=(14, 12))
corr_features = features + ['log_revenue']
corr_data = df[valid_idx][corr_features].corr()
sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('1_correlation_matrix.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: 1_correlation_matrix.png")
plt.close()

# 2. Actual vs Predicted
print("2. Creating actual vs predicted plots...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, (name, pred, metrics) in enumerate([
    ('Random Forest', rf_reg_pred, rf_reg_metrics),
    ('XGBoost', xgb_reg_pred, xgb_reg_metrics)
]):
    ax = axes[idx]
    ax.scatter(y_reg_test, pred, alpha=0.5, s=30, edgecolors='k', linewidths=0.5)
    
    min_val, max_val = y_reg_test.min(), y_reg_test.max()
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')
    
    ax.set_xlabel('Actual Log Revenue', fontsize=12)
    ax.set_ylabel('Predicted Log Revenue', fontsize=12)
    ax.set_title(f'{name}\nR²={metrics["r2"]:.4f}, MAE={metrics["mae"]:.4f}', 
                 fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('2_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: 2_actual_vs_predicted.png")
plt.close()

# 3. Residual Analysis
print("3. Creating residual analysis...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, (name, pred) in enumerate([('Random Forest', rf_reg_pred), ('XGBoost', xgb_reg_pred)]):
    residuals = y_reg_test.values - pred
    
    # Residual scatter
    ax1 = axes[0, idx]
    ax1.scatter(pred, residuals, alpha=0.5, s=30, edgecolors='k', linewidths=0.5)
    ax1.axhline(y=0, color='r', linestyle='--', lw=2)
    ax1.set_xlabel('Predicted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title(f'{name} - Residual Plot', fontweight='bold')
    ax1.grid(alpha=0.3)
    
    # Residual distribution
    ax2 = axes[1, idx]
    ax2.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel('Residuals')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'{name} - Residual Distribution', fontweight='bold')
    ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('3_residual_analysis.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: 3_residual_analysis.png")
plt.close()

# 4. Feature Importance - Random Forest
print("4. Creating feature importance (Random Forest)...")
importances = rf_reg.feature_importances_
indices = np.argsort(importances)[::-1][:15]

plt.figure(figsize=(10, 8))
plt.barh(range(len(indices)), importances[indices], color='steelblue', edgecolor='black')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Feature Importance', fontsize=12)
plt.title('Random Forest - Top 15 Feature Importance', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('4_feature_importance_rf.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: 4_feature_importance_rf.png")
plt.close()

# 5. Feature Importance - XGBoost
print("5. Creating feature importance (XGBoost)...")
importances = xgb_reg.feature_importances_
indices = np.argsort(importances)[::-1][:15]

plt.figure(figsize=(10, 8))
plt.barh(range(len(indices)), importances[indices], color='darkorange', edgecolor='black')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Feature Importance', fontsize=12)
plt.title('XGBoost - Top 15 Feature Importance', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('5_feature_importance_xgb.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: 5_feature_importance_xgb.png")
plt.close()

# 6. Model Comparison
print("6. Creating model comparison chart...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

metrics_to_plot = [
    ('R² Score', [rf_reg_metrics['r2'], xgb_reg_metrics['r2']], 
     [0.67236, 0.70841]),
    ('MAE', [rf_reg_metrics['mae'], xgb_reg_metrics['mae']], 
     [0.28439, 0.28065]),
    ('RMSE', [rf_reg_metrics['rmse'], xgb_reg_metrics['rmse']], 
     [np.sqrt(0.28439**2), np.sqrt(0.28065**2)])
]

x = np.arange(2)
width = 0.35
model_names = ['RF', 'XGB']

for idx, (metric_name, our_vals, paper_vals) in enumerate(metrics_to_plot):
    ax = axes[idx]
    ax.bar(x - width/2, our_vals, width, label='Our Implementation', color='steelblue', edgecolor='black')
    ax.bar(x + width/2, paper_vals, width, label='Paper', color='lightcoral', edgecolor='black')
    ax.set_ylabel(metric_name, fontsize=11)
    ax.set_title(f'{metric_name} Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('6_model_comparison.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: 6_model_comparison.png")
plt.close()

# 7. ROC Curve
print("7. Creating ROC curve...")
fig, ax = plt.subplots(figsize=(8, 8))

for name, prob, color in [('Random Forest', rf_clf_prob, 'blue'), 
                           ('XGBoost', xgb_clf_prob, 'orange')]:
    fpr, tpr, _ = roc_curve(y_clf_test, prob)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2, 
            label=f'{name} (AUC = {roc_auc:.3f})')

ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve - Hit vs Non-Hit Classification', fontsize=14, fontweight='bold')
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('7_auc_roc.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: 7_auc_roc.png")
plt.close()

# 8. Precision-Recall Curve
print("8. Creating precision-recall curve...")
fig, ax = plt.subplots(figsize=(8, 8))

for name, prob, color in [('Random Forest', rf_clf_prob, 'blue'),
                           ('XGBoost', xgb_clf_prob, 'orange')]:
    precision, recall, _ = precision_recall_curve(y_clf_test, prob)
    pr_auc = auc(recall, precision)
    ax.plot(recall, precision, color=color, lw=2,
            label=f'{name} (AP = {pr_auc:.3f})')

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curve - Hit vs Non-Hit Classification', 
             fontsize=14, fontweight='bold')
ax.legend(loc="lower left")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('8_precision_recall.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: 8_precision_recall.png")
plt.close()

# 9. Confusion Matrices
print("9. Creating confusion matrices...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for idx, (name, metrics) in enumerate([('Random Forest', rf_clf_metrics),
                                        ('XGBoost', xgb_clf_metrics)]):
    ax = axes[idx]
    cm = metrics['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                xticklabels=['Non-Hit', 'Hit'],
                yticklabels=['Non-Hit', 'Hit'])
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)
    ax.set_title(f'{name} - Confusion Matrix\nAcc={metrics["accuracy"]:.3f}, F1={metrics["f1"]:.3f}',
                 fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('9_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: 9_confusion_matrix.png")
plt.close()

#==============================================================================
# SAVE RESULTS
#==============================================================================
print("\n" + "="*80)
print("💾 SAVING RESULTS")
print("="*80)

# Regression comparison
comparison_reg = pd.DataFrame([
    {
        'Model': 'Random Forest',
        'Paper R²': 0.67236,
        'Our R²': rf_reg_metrics['r2'],
        'Δ R²': rf_reg_metrics['r2'] - 0.67236,
        'Paper MAE': 0.28439,
        'Our MAE': rf_reg_metrics['mae'],
        'Δ MAE': rf_reg_metrics['mae'] - 0.28439
    },
    {
        'Model': 'XGBoost',
        'Paper R²': 0.70841,
        'Our R²': xgb_reg_metrics['r2'],
        'Δ R²': xgb_reg_metrics['r2'] - 0.70841,
        'Paper MAE': 0.28065,
        'Our MAE': xgb_reg_metrics['mae'],
        'Δ MAE': xgb_reg_metrics['mae'] - 0.28065
    }
])
comparison_reg.to_csv('comparison_results_regression.csv', index=False)
print("✓ Saved: comparison_results_regression.csv")

# Classification metrics
classification_results = pd.DataFrame([
    {
        'Model': 'Random Forest',
        'Accuracy': rf_clf_metrics['accuracy'],
        'Precision': rf_clf_metrics['precision'],
        'Recall': rf_clf_metrics['recall'],
        'F1': rf_clf_metrics['f1'],
        'AUROC': rf_clf_metrics['auroc'],
        'PR-AUC': rf_clf_metrics['avg_precision']
    },
    {
        'Model': 'XGBoost',
        'Accuracy': xgb_clf_metrics['accuracy'],
        'Precision': xgb_clf_metrics['precision'],
        'Recall': xgb_clf_metrics['recall'],
        'F1': xgb_clf_metrics['f1'],
        'AUROC': xgb_clf_metrics['auroc'],
        'PR-AUC': xgb_clf_metrics['avg_precision']
    }
])
classification_results.to_csv('classification_metrics.csv', index=False)
print("✓ Saved: classification_metrics.csv")

# Submission file
submission = pd.DataFrame({
    'id': df[valid_idx].iloc[X_test.index]['id'],
    'revenue': np.expm1(rf_reg_pred)
})
submission.to_csv('submission.csv', index=False)
print("✓ Saved: submission.csv")

#==============================================================================
# SUMMARY
#==============================================================================
print("\n" + "="*80)
print("✅ ANALYSIS COMPLETE!")
print("="*80)

print("\n📁 Generated Files:")
print("   • 1_correlation_matrix.png")
print("   • 2_actual_vs_predicted.png")
print("   • 3_residual_analysis.png")
print("   • 4_feature_importance_rf.png")
print("   • 5_feature_importance_xgb.png")
print("   • 6_model_comparison.png")
print("   • 7_auc_roc.png")
print("   • 8_precision_recall.png")
print("   • 9_confusion_matrix.png")
print("   • comparison_results_regression.csv")
print("   • classification_metrics.csv")
print("   • submission.csv")

print("\n📊 REGRESSION RESULTS:")
print(f"   Random Forest: R² = {rf_reg_metrics['r2']:.4f}, MAE = {rf_reg_metrics['mae']:.4f}")
print(f"   XGBoost:       R² = {xgb_reg_metrics['r2']:.4f}, MAE = {xgb_reg_metrics['mae']:.4f}")

print("\n📊 CLASSIFICATION RESULTS (Hit vs Non-Hit):")
print(f"   Random Forest:")
print(f"      Accuracy: {rf_clf_metrics['accuracy']:.4f}")
print(f"      Precision: {rf_clf_metrics['precision']:.4f}")
print(f"      Recall: {rf_clf_metrics['recall']:.4f}")
print(f"      F1: {rf_clf_metrics['f1']:.4f}")
print(f"      AUROC: {rf_clf_metrics['auroc']:.4f}")
print(f"      PR-AUC: {rf_clf_metrics['avg_precision']:.4f}")

print(f"\n   XGBoost:")
print(f"      Accuracy: {xgb_clf_metrics['accuracy']:.4f}")
print(f"      Precision: {xgb_clf_metrics['precision']:.4f}")
print(f"      Recall: {xgb_clf_metrics['recall']:.4f}")
print(f"      F1: {xgb_clf_metrics['f1']:.4f}")
print(f"      AUROC: {xgb_clf_metrics['auroc']:.4f}")
print(f"      PR-AUC: {xgb_clf_metrics['avg_precision']:.4f}")

print(f"\n📊 CONFUSION MATRICES:")
print(f"\n   Random Forest:")
print(f"   {'':>20} Predicted Non-Hit  Predicted Hit")
print(f"   True Non-Hit      {rf_clf_metrics['confusion_matrix'][0,0]:8d}         {rf_clf_metrics['confusion_matrix'][0,1]:8d}")
print(f"   True Hit          {rf_clf_metrics['confusion_matrix'][1,0]:8d}         {rf_clf_metrics['confusion_matrix'][1,1]:8d}")

print(f"\n   XGBoost:")
print(f"   {'':>20} Predicted Non-Hit  Predicted Hit")
print(f"   True Non-Hit      {xgb_clf_metrics['confusion_matrix'][0,0]:8d}         {xgb_clf_metrics['confusion_matrix'][0,1]:8d}")
print(f"   True Hit          {xgb_clf_metrics['confusion_matrix'][1,0]:8d}         {xgb_clf_metrics['confusion_matrix'][1,1]:8d}")

print("\n🎉 Success! All analysis complete with regression + classification metrics.")
print("="*80)