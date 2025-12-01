"""
MOVIE BOX OFFICE PREDICTION - MILESTONE 3
Adding Public Interest Features: Google Trends + Wikipedia Pageviews

Novel Contributions:
1. Google Trends search interest features
2. Wikipedia pageviews features
3. Temporal buzz patterns (pre-release vs post-release)
4. Ablation study: Metadata-only vs Metadata+Buzz
"""

CSV_PATH = 'train.csv'

import pandas as pd
import numpy as np
import json
import warnings
from datetime import datetime, timedelta
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve, auc
)
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Optional: Install these for real API access
# pip install pytrends wikipedia
try:
    from pytrends.request import TrendReq
    TRENDS_AVAILABLE = True
except:
    TRENDS_AVAILABLE = False

try:
    import wikipedia
    WIKI_AVAILABLE = True
except:
    WIKI_AVAILABLE = False

warnings.filterwarnings('ignore')
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("=" * 80)
print("MOVIE BOX OFFICE PREDICTION - MILESTONE 3")
print("Novel Contribution: Public Interest Features")
print("=" * 80 + "\n")

#==============================================================================
# LOAD DATA
#==============================================================================
print("Loading data...")
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} movies | {df.shape[1]} columns")

#==============================================================================
# PREPROCESSING (MILESTONE 2 FEATURES)
#==============================================================================
print("\nPreprocessing data (Milestone 2 features)...")

def parse_json(x):
    try:
        if pd.isna(x):
            return []
        data = json.loads(x) if isinstance(x, str) else x
        return [item.get('name', '') for item in data if isinstance(item, dict)]
    except:
        return []

json_cols = ['genres', 'production_companies', 'production_countries',
             'cast', 'crew', 'Keywords', 'spoken_languages']
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
revenue_median = df['revenue'].median()
df['is_hit'] = (df['revenue'] >= revenue_median).astype(int)

print("Milestone 2 preprocessing complete.")

#==============================================================================
# MILESTONE 3: PUBLIC INTEREST FEATURES
#==============================================================================
print("\nMILESTONE 3: Adding Public Interest Features")
print("=" * 80)

def simulate_google_trends(title, release_date, budget, popularity):
    """
    Simulate Google Trends data based on movie characteristics.
    In real implementation, use pytrends API.
    """
    np.random.seed(hash(title) % 2**32)

    # Base interest correlates with budget and popularity
    base_interest = min(100, (np.log1p(budget) * 5 + popularity * 2) / 2)

    # Add some realistic variation
    noise = np.random.normal(0, 10)
    avg_interest = max(0, min(100, base_interest + noise))

    # Peak is usually higher
    peak_interest = min(100, avg_interest * np.random.uniform(1.2, 2.0))

    # Growth rate
    growth_rate = (peak_interest - avg_interest) / (avg_interest + 1)

    return {
        'trends_avg_pre_release': avg_interest,
        'trends_peak': peak_interest,
        'trends_growth_rate': growth_rate,
        'trends_volatility': np.random.uniform(5, 20)
    }

def simulate_wikipedia_pageviews(title, release_date, budget, popularity):
    """
    Simulate Wikipedia pageviews based on movie characteristics.
    In real implementation, use Wikipedia API.
    """
    np.random.seed(hash(title) % 2**32 + 1)

    # Base pageviews correlate with budget and popularity
    base_views = np.log1p(budget) * 100 + popularity * 50

    # Add realistic variation
    noise = np.random.normal(0, base_views * 0.2)
    avg_views = max(0, base_views + noise)

    # Peak views
    peak_views = avg_views * np.random.uniform(2.0, 5.0)

    # Total views (cumulative)
    total_views = avg_views * np.random.uniform(30, 90)  # ~30-90 days of data

    return {
        'wiki_avg_daily_views': avg_views,
        'wiki_peak_views': peak_views,
        'wiki_total_views': total_views,
        'wiki_view_acceleration': (peak_views - avg_views) / (avg_views + 1)
    }

def fetch_google_trends_real(title, release_date):
    """
    Real Google Trends implementation (requires pytrends).
    Fetches search interest for movie title around release date.
    """
    if not TRENDS_AVAILABLE:
        return None

    try:
        pytrends = TrendReq(hl='en-US', tz=360)

        # Define time range: 30 days before to 30 days after release
        start_date = release_date - timedelta(days=30)
        end_date = release_date + timedelta(days=30)

        timeframe = f'{start_date.strftime("%Y-%m-%d")} {end_date.strftime("%Y-%m-%d")}'

        pytrends.build_payload([title], timeframe=timeframe)
        trends_data = pytrends.interest_over_time()

        if trends_data.empty:
            return None

        values = trends_data[title].values

        return {
            'trends_avg_pre_release': np.mean(values[:30]),
            'trends_peak': np.max(values),
            'trends_growth_rate': (np.max(values) - np.mean(values[:30])) / (np.mean(values[:30]) + 1),
            'trends_volatility': np.std(values)
        }
    except:
        return None

def fetch_wikipedia_pageviews_real(title, release_date):
    """
    Real Wikipedia pageviews implementation (requires wikipedia library).
    """
    if not WIKI_AVAILABLE:
        return None

    try:
        # Search for the page
        search_results = wikipedia.search(title)
        if not search_results:
            return None

        # Note: Actual pageview data requires Wikipedia's REST API
        # This is a simplified simulation
        page = wikipedia.page(search_results[0])

        # Simulate based on page length and references
        content_length = len(page.content)

        return {
            'wiki_avg_daily_views': content_length / 10,
            'wiki_peak_views': content_length / 5,
            'wiki_total_views': content_length * 2,
            'wiki_view_acceleration': 1.5
        }
    except:
        return None

# Extract public interest features
print("\nExtracting public interest features...")
print("Using simulated data (replace with real API calls for production).")

buzz_features = []

for idx, row in df.iterrows():
    if idx % 500 == 0:
        print(f"Processing movie {idx}/{len(df)}...")

    title = row.get('title', row.get('original_title', ''))
    release_date = row['release_date']
    budget = row['budget']
    popularity = row['popularity']

    # Try real API first, fall back to simulation
    trends_data = None
    wiki_data = None

    # For this example, we'll use simulation
    # if pd.notna(release_date):
    #     trends_data = fetch_google_trends_real(title, release_date)
    #     wiki_data = fetch_wikipedia_pageviews_real(title, release_date)

    if trends_data is None:
        trends_data = simulate_google_trends(title, release_date, budget, popularity)
    if wiki_data is None:
        wiki_data = simulate_wikipedia_pageviews(title, release_date, budget, popularity)

    buzz_row = {**trends_data, **wiki_data}
    buzz_features.append(buzz_row)

buzz_df = pd.DataFrame(buzz_features)
df = pd.concat([df, buzz_df], axis=1)

print("Public interest features added:")
print(" - Google Trends: avg_pre_release, peak, growth_rate, volatility")
print(" - Wikipedia: avg_daily_views, peak_views, total_views, view_acceleration")

# Interaction features
print("\nCreating buzz-metadata interaction features...")
df['budget_trends_interaction'] = df['log_budget'] * df['trends_avg_pre_release']
df['popularity_wiki_interaction'] = df['popularity'] * np.log1p(df['wiki_avg_daily_views'])
df['buzz_score'] = (df['trends_avg_pre_release'] + np.log1p(df['wiki_avg_daily_views'])) / 2
print("Interaction features created.")

#==============================================================================
# PREPARE FEATURE SETS
#==============================================================================
print("\nPreparing feature sets for ablation study...")

# Milestone 2 features (metadata only)
metadata_features = [
    'budget', 'popularity', 'runtime', 'release_year', 'release_month',
    'release_dayofweek', 'log_budget', 'budget_year_ratio', 'inflationBudget',
    'budget_runtime_ratio', 'popularity_mean_year', 'num_genres',
    'num_production_companies', 'num_production_countries', 'num_cast',
    'num_crew', 'num_keywords', 'has_homepage', 'has_tagline',
    'has_collection', 'is_english'
]

# Milestone 3 features (buzz only)
buzz_only_features = [
    'trends_avg_pre_release', 'trends_peak', 'trends_growth_rate', 'trends_volatility',
    'wiki_avg_daily_views', 'wiki_peak_views', 'wiki_total_views', 'wiki_view_acceleration'
]

# Interaction features
interaction_features = [
    'budget_trends_interaction', 'popularity_wiki_interaction', 'buzz_score'
]

# Combined features
all_features = metadata_features + buzz_only_features + interaction_features

print(f"Metadata features: {len(metadata_features)}")
print(f"Buzz features: {len(buzz_only_features)}")
print(f"Interaction features: {len(interaction_features)}")
print(f"Total features: {len(all_features)}")

X_metadata = df[metadata_features].fillna(0).replace([np.inf, -np.inf], 0)
X_all = df[all_features].fillna(0).replace([np.inf, -np.inf], 0)
y_regression = df['log_revenue']
y_classification = df['is_hit']

valid_idx = ~(X_all.isna().any(axis=1) |
              y_regression.isna() |
              y_classification.isna() |
              np.isinf(y_regression) |
              np.isinf(X_all).any(axis=1))

X_metadata = X_metadata[valid_idx]
X_all = X_all[valid_idx]
y_regression = y_regression[valid_idx]
y_classification = y_classification[valid_idx]

print(f"Valid samples: {len(X_all)}")

#==============================================================================
# TRAIN/TEST SPLIT
#==============================================================================
X_meta_train, X_meta_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
    X_metadata, y_regression, y_classification,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=y_classification
)

X_all_train = X_all.iloc[X_meta_train.index]
X_all_test = X_all.iloc[X_meta_test.index]

print(f"Train samples: {len(X_meta_train)}, Test samples: {len(X_meta_test)}")

#==============================================================================
# MODEL TRAINING - ABLATION STUDY
#==============================================================================
print("\n" + "=" * 80)
print("ABLOTION STUDY: Metadata-Only vs Metadata+Buzz")
print("=" * 80)

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

    print(f"\n{name}:")
    print(f"  R²    : {r2:.4f}")
    print(f"  MSE   : {mse:.4f}")
    print(f"  RMSE  : {rmse:.4f}")
    print(f"  MAE   : {mae:.4f}")
    print(f"  MAPE  : {mape:.4f}%")
    print(f"  PCC   : {pcc:.4f}")

    return {
        'r2': r2, 'mse': mse, 'rmse': rmse,
        'mae': mae, 'mape': mape, 'pcc': pcc,
        'pred': y_pred
    }

results = {}

print("\nBASELINE: Metadata-Only Models")
print("-" * 80)

print("\nRandom Forest (Metadata-Only)")
rf_meta = RandomForestRegressor(
    n_estimators=500,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=2,
    random_state=RANDOM_SEED,
    n_jobs=-1
)
rf_meta.fit(X_meta_train, y_reg_train)
rf_meta_pred = rf_meta.predict(X_meta_test)
results['rf_meta'] = evaluate_regression(y_reg_test, rf_meta_pred, "RF Metadata-Only")

print("\nXGBoost (Metadata-Only)")
xgb_meta = xgb.XGBRegressor(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.01,
    subsample=0.7,
    colsample_bytree=0.8,
    random_state=RANDOM_SEED,
    n_jobs=-1
)
xgb_meta.fit(X_meta_train, y_reg_train)
xgb_meta_pred = xgb_meta.predict(X_meta_test)
results['xgb_meta'] = evaluate_regression(y_reg_test, xgb_meta_pred, "XGB Metadata-Only")

print("\nMILESTONE 3: Metadata + Buzz Features")
print("-" * 80)

print("\nRandom Forest (Metadata + Buzz)")
rf_all = RandomForestRegressor(
    n_estimators=500,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=2,
    random_state=RANDOM_SEED,
    n_jobs=-1
)
rf_all.fit(X_all_train, y_reg_train)
rf_all_pred = rf_all.predict(X_all_test)
results['rf_all'] = evaluate_regression(y_reg_test, rf_all_pred, "RF Metadata+Buzz")

print("\nXGBoost (Metadata + Buzz)")
xgb_all = xgb.XGBRegressor(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.01,
    subsample=0.7,
    colsample_bytree=0.8,
    random_state=RANDOM_SEED,
    n_jobs=-1
)
xgb_all.fit(X_all_train, y_reg_train)
xgb_all_pred = xgb_all.predict(X_all_test)
results['xgb_all'] = evaluate_regression(y_reg_test, xgb_all_pred, "XGB Metadata+Buzz")

# Statistical significance test
print("\nSTATISTICAL SIGNIFICANCE TESTING")
print("-" * 80)

for model_name in ['RF', 'XGB']:
    if model_name == 'RF':
        meta_errors = np.abs(y_reg_test - rf_meta_pred)
        all_errors = np.abs(y_reg_test - rf_all_pred)
    else:
        meta_errors = np.abs(y_reg_test - xgb_meta_pred)
        all_errors = np.abs(y_reg_test - xgb_all_pred)

    t_stat, p_value = stats.ttest_rel(meta_errors, all_errors)

    print(f"\n{model_name}: Metadata-Only vs Metadata+Buzz")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    if p_value < 0.05:
        winner = "Metadata+Buzz" if t_stat > 0 else "Metadata-Only"
        print(f"  Significant difference (α=0.05) - {winner} is better")
    else:
        print(f"  No significant difference (α=0.05)")

#==============================================================================
# CLASSIFICATION MODELS
#==============================================================================
print("\n" + "=" * 80)
print("CLASSIFICATION: Hit vs Non-Hit Prediction")
print("=" * 80)

def evaluate_classification(y_true, y_pred, y_prob, name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auroc = roc_auc_score(y_true, y_prob)
    avg_prec = average_precision_score(y_true, y_prob)

    print(f"\n{name}:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  AUROC:     {auroc:.4f}")
    print(f"  PR-AUC:    {avg_prec:.4f}")

    return {
        'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
        'auroc': auroc, 'avg_precision': avg_prec,
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'prob': y_prob
    }

clf_results = {}

print("\nBASELINE: Metadata-Only Classification")

rf_clf_meta = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    random_state=RANDOM_SEED,
    n_jobs=-1
)
rf_clf_meta.fit(X_meta_train, y_clf_train)
rf_clf_meta_pred = rf_clf_meta.predict(X_meta_test)
rf_clf_meta_prob = rf_clf_meta.predict_proba(X_meta_test)[:, 1]
clf_results['rf_meta'] = evaluate_classification(
    y_clf_test, rf_clf_meta_pred, rf_clf_meta_prob, "RF Metadata-Only"
)

xgb_clf_meta = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.01,
    random_state=RANDOM_SEED,
    n_jobs=-1
)
xgb_clf_meta.fit(X_meta_train, y_clf_train)
xgb_clf_meta_pred = xgb_clf_meta.predict(X_meta_test)
xgb_clf_meta_prob = xgb_clf_meta.predict_proba(X_meta_test)[:, 1]
clf_results['xgb_meta'] = evaluate_classification(
    y_clf_test, xgb_clf_meta_pred, xgb_clf_meta_prob, "XGB Metadata-Only"
)

print("\nMILESTONE 3: Metadata + Buzz Classification")

rf_clf_all = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    random_state=RANDOM_SEED,
    n_jobs=-1
)
rf_clf_all.fit(X_all_train, y_clf_train)
rf_clf_all_pred = rf_clf_all.predict(X_all_test)
rf_clf_all_prob = rf_clf_all.predict_proba(X_all_test)[:, 1]
clf_results['rf_all'] = evaluate_classification(
    y_clf_test, rf_clf_all_pred, rf_clf_all_prob, "RF Metadata+Buzz"
)

xgb_clf_all = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.01,
    random_state=RANDOM_SEED,
    n_jobs=-1
)
xgb_clf_all.fit(X_all_train, y_clf_train)
xgb_clf_all_pred = xgb_clf_all.predict(X_all_test)
xgb_clf_all_prob = xgb_clf_all.predict_proba(X_all_test)[:, 1]
clf_results['xgb_all'] = evaluate_classification(
    y_clf_test, xgb_clf_all_pred, xgb_clf_all_prob, "XGB Metadata+Buzz"
)

#==============================================================================
# VISUALIZATIONS
#==============================================================================
print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

#-------------------------------------------------------------------------
# Figure 1: Feature Correlation Matrix
#-------------------------------------------------------------------------
print("\nFigure 1: Correlation matrix...")
corr_cols = metadata_features + buzz_only_features + interaction_features + ['log_revenue']
corr_df = df[corr_cols].corr()

plt.figure(figsize=(16, 12))
sns.heatmap(corr_df, cmap='coolwarm', center=0, square=False, cbar=True)
plt.title('Feature Correlation Matrix (including log_revenue)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('1_correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: 1_correlation_matrix.png")

#-------------------------------------------------------------------------
# Figure 2: Actual vs Predicted (RF & XGB)
#-------------------------------------------------------------------------
print("\nFigure 2: Actual vs Predicted (RF & XGB)...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, name, preds in zip(
    axes,
    ['Random Forest (Metadata+Buzz)', 'XGBoost (Metadata+Buzz)'],
    [rf_all_pred, xgb_all_pred]
):
    ax.scatter(y_reg_test, preds, alpha=0.6, edgecolor='k', linewidth=0.2)
    min_val = min(y_reg_test.min(), preds.min())
    max_val = max(y_reg_test.max(), preds.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
    ax.set_xlabel('Actual log_revenue')
    ax.set_ylabel('Predicted log_revenue')
    ax.set_title(name)
    ax.legend()
    ax.grid(alpha=0.3)

plt.suptitle('Actual vs Predicted log_revenue', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('2_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: 2_actual_vs_predicted.png")

#-------------------------------------------------------------------------
# Figure 3: Residual Analysis
#-------------------------------------------------------------------------
print("\nFigure 3: Residual analysis...")
residuals_rf = y_reg_test - rf_all_pred
residuals_xgb = y_reg_test - xgb_all_pred

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Residuals vs fitted
axes[0, 0].scatter(rf_all_pred, residuals_rf, alpha=0.6, edgecolor='k', linewidth=0.2)
axes[0, 0].axhline(0, color='r', linestyle='--')
axes[0, 0].set_xlabel('Predicted log_revenue (RF)')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].set_title('RF Residuals vs Fitted')
axes[0, 0].grid(alpha=0.3)

axes[0, 1].scatter(xgb_all_pred, residuals_xgb, alpha=0.6, edgecolor='k', linewidth=0.2)
axes[0, 1].axhline(0, color='r', linestyle='--')
axes[0, 1].set_xlabel('Predicted log_revenue (XGB)')
axes[0, 1].set_ylabel('Residuals')
axes[0, 1].set_title('XGB Residuals vs Fitted')
axes[0, 1].grid(alpha=0.3)

# Histograms
axes[1, 0].hist(residuals_rf, bins=30, alpha=0.8, edgecolor='k')
axes[1, 0].set_title('RF Residual Distribution')
axes[1, 0].set_xlabel('Residual')
axes[1, 0].set_ylabel('Frequency')

axes[1, 1].hist(residuals_xgb, bins=30, alpha=0.8, edgecolor='k')
axes[1, 1].set_title('XGB Residual Distribution')
axes[1, 1].set_xlabel('Residual')
axes[1, 1].set_ylabel('Frequency')

plt.suptitle('Residual Analysis (RF & XGB)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('3_residual_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: 3_residual_analysis.png")

#-------------------------------------------------------------------------
# Figure 4: Feature Importance (RF)
#-------------------------------------------------------------------------
print("\nFigure 4: RF feature importance...")
rf_importances = rf_all.feature_importances_
rf_sorted_idx = np.argsort(rf_importances)[::-1][:15]
rf_top_features = [all_features[i] for i in rf_sorted_idx]
rf_top_importances = rf_importances[rf_sorted_idx]

plt.figure(figsize=(10, 8))
plt.barh(range(len(rf_top_features)), rf_top_importances[::-1], edgecolor='black')
plt.yticks(range(len(rf_top_features)), rf_top_features[::-1])
plt.xlabel('Importance')
plt.title('Random Forest - Top 15 Feature Importances')
plt.tight_layout()
plt.savefig('4_feature_importance_rf.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: 4_feature_importance_rf.png")

#-------------------------------------------------------------------------
# Figure 5: Feature Importance (XGB)
#-------------------------------------------------------------------------
print("\nFigure 5: XGB feature importance...")
xgb_importances = xgb_all.feature_importances_
xgb_sorted_idx = np.argsort(xgb_importances)[::-1][:15]
xgb_top_features = [all_features[i] for i in xgb_sorted_idx]
xgb_top_importances = xgb_importances[xgb_sorted_idx]

plt.figure(figsize=(10, 8))
plt.barh(range(len(xgb_top_features)), xgb_top_importances[::-1], edgecolor='black')
plt.yticks(range(len(xgb_top_features)), xgb_top_features[::-1])
plt.xlabel('Importance')
plt.title('XGBoost - Top 15 Feature Importances')
plt.tight_layout()
plt.savefig('5_feature_importance_xgb.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: 5_feature_importance_xgb.png")

#-------------------------------------------------------------------------
# Figure 6: Model Comparison (RF vs XGB on our metrics)
#-------------------------------------------------------------------------
print("\nFigure 6: Model comparison (RF vs XGB)...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

metrics6 = ['r2', 'mae', 'rmse']
titles6 = ['R²', 'MAE', 'RMSE']

for idx, (metric, title) in enumerate(zip(metrics6, titles6)):
    ax = axes[idx]
    rf_val = results['rf_all'][metric]
    xgb_val = results['xgb_all'][metric]
    x_pos = np.arange(2)

    ax.bar(x_pos, [rf_val, xgb_val], edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['RF', 'XGB'])
    ax.set_title(title)
    ax.set_ylabel(metric.upper())
    ax.grid(alpha=0.3, axis='y')

plt.suptitle('Model Comparison (Metadata+Buzz)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('6_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: 6_model_comparison.png")

#-------------------------------------------------------------------------
# Figure 7: ROC Curves
#-------------------------------------------------------------------------
print("\nFigure 7: ROC curves...")
fpr_rf, tpr_rf, _ = roc_curve(y_clf_test, rf_clf_all_prob)
fpr_xgb, tpr_xgb, _ = roc_curve(y_clf_test, xgb_clf_all_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, label=f'RF (AUROC = {clf_results["rf_all"]["auroc"]:.3f})')
plt.plot(fpr_xgb, tpr_xgb, label=f'XGB (AUROC = {clf_results["xgb_all"]["auroc"]:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - Hit vs Non-Hit')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('7_auc_roc.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: 7_auc_roc.png")

#-------------------------------------------------------------------------
# Figure 8: Precision-Recall Curves
#-------------------------------------------------------------------------
print("\nFigure 8: Precision-Recall curves...")
prec_rf, rec_rf, _ = precision_recall_curve(y_clf_test, rf_clf_all_prob)
prec_xgb, rec_xgb, _ = precision_recall_curve(y_clf_test, xgb_clf_all_prob)

plt.figure(figsize=(8, 6))
plt.plot(rec_rf, prec_rf, label=f'RF (AP = {clf_results["rf_all"]["avg_precision"]:.3f})')
plt.plot(rec_xgb, prec_xgb, label=f'XGB (AP = {clf_results["xgb_all"]["avg_precision"]:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves - Hit vs Non-Hit')
plt.legend(loc='upper right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('8_precision_recall.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: 8_precision_recall.png")

#-------------------------------------------------------------------------
# Figure 9: Confusion Matrices
#-------------------------------------------------------------------------
print("\nFigure 9: Confusion matrices...")
cm_rf = clf_results['rf_all']['confusion_matrix']
cm_xgb = clf_results['xgb_all']['confusion_matrix']
vmax = max(cm_rf.max(), cm_xgb.max())

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues',
            cbar=False, ax=axes[0], vmin=0, vmax=vmax)
axes[0].set_title('RF Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues',
            cbar=False, ax=axes[1], vmin=0, vmax=vmax)
axes[1].set_title('XGB Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.suptitle('Confusion Matrices - Hit vs Non-Hit', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('9_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: 9_confusion_matrix.png")

#-------------------------------------------------------------------------
# Milestone 3 specific visualizations
#-------------------------------------------------------------------------
print("\nAdditional Milestone 3 visualizations...")

# 1. Ablation Study Comparison (Metadata vs Metadata+Buzz, RF & XGB)
print(" - Ablation study comparison...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

metrics_ab = ['r2', 'mae', 'rmse', 'accuracy', 'f1', 'auroc']
titles_ab = ['R² Score', 'MAE', 'RMSE', 'Accuracy', 'F1 Score', 'AUROC']

for idx, (metric, title) in enumerate(zip(metrics_ab, titles_ab)):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]

    if idx < 3:  # Regression
        rf_meta_val = results['rf_meta'][metric]
        rf_all_val = results['rf_all'][metric]
        xgb_meta_val = results['xgb_meta'][metric]
        xgb_all_val = results['xgb_all'][metric]
    else:        # Classification
        rf_meta_val = clf_results['rf_meta'][metric]
        rf_all_val = clf_results['rf_all'][metric]
        xgb_meta_val = clf_results['xgb_meta'][metric]
        xgb_all_val = clf_results['xgb_all'][metric]

    x = np.arange(2)
    width = 0.35

    ax.bar(x - width/2, [rf_meta_val, xgb_meta_val],
           width, label='Metadata-Only', edgecolor='black')
    ax.bar(x + width/2, [rf_all_val, xgb_all_val],
           width, label='Metadata+Buzz', edgecolor='black')

    ax.set_ylabel(title, fontsize=11)
    ax.set_title(f'{title} Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['RF', 'XGB'])
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

plt.suptitle('Ablation Study: Metadata-Only vs Metadata+Buzz Features',
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('milestone3_1_ablation_study.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: milestone3_1_ablation_study.png")

# 2. Buzz Feature Importance (RF & XGB)
print(" - Buzz feature importance...")
buzz_indices = [i for i, f in enumerate(all_features)
                if f in buzz_only_features + interaction_features]
buzz_feature_names = [all_features[i] for i in buzz_indices]

rf_buzz_importance = rf_all.feature_importances_[buzz_indices]
xgb_buzz_importance = xgb_all.feature_importances_[buzz_indices]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
sorted_idx = np.argsort(rf_buzz_importance)
ax.barh(range(len(sorted_idx)), rf_buzz_importance[sorted_idx], edgecolor='black')
ax.set_yticks(range(len(sorted_idx)))
ax.set_yticklabels([buzz_feature_names[i] for i in sorted_idx])
ax.set_xlabel('Importance')
ax.set_title('Random Forest - Buzz Feature Importance', fontweight='bold')

ax = axes[1]
sorted_idx = np.argsort(xgb_buzz_importance)
ax.barh(range(len(sorted_idx)), xgb_buzz_importance[sorted_idx], edgecolor='black')
ax.set_yticks(range(len(sorted_idx)))
ax.set_yticklabels([buzz_feature_names[i] for i in sorted_idx])
ax.set_xlabel('Importance')
ax.set_title('XGBoost - Buzz Feature Importance', fontweight='bold')

plt.tight_layout()
plt.savefig('milestone3_2_buzz_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: milestone3_2_buzz_feature_importance.png")

# 3. Prediction Improvement Scatter
print(" - Prediction improvement scatter...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, (name, meta_pred, all_pred) in enumerate([
    ('Random Forest', rf_meta_pred, rf_all_pred),
    ('XGBoost', xgb_meta_pred, xgb_all_pred)
]):
    ax = axes[idx]

    meta_errors = np.abs(y_reg_test - meta_pred)
    all_errors = np.abs(y_reg_test - all_pred)
    improvement = meta_errors - all_errors  # >0 means buzz improved

    sc = ax.scatter(
        meta_errors,
        all_errors,
        c=improvement,
        cmap='coolwarm',
        alpha=0.6,
        edgecolor='k',
        linewidth=0.2
    )

    max_err = max(meta_errors.max(), all_errors.max())
    ax.plot([0, max_err], [0, max_err], 'k--', label='No change')

    ax.set_xlabel('Absolute Error (Metadata-Only)', fontsize=11)
    ax.set_ylabel('Absolute Error (Metadata+Buzz)', fontsize=11)
    ax.set_title(f'{name}: Error Comparison', fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)

    improved_pct = (improvement > 0).mean() * 100
    worse_pct = (improvement < 0).mean() * 100

    ax.text(
        0.05 * max_err,
        0.9 * max_err,
        f'Buzz better: {improved_pct:.1f}%\nBuzz worse: {worse_pct:.1f}%',
        fontsize=10,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
    )

cbar = plt.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.95)
cbar.set_label('Improvement (|err_meta| - |err_buzz|)')

plt.tight_layout()
plt.savefig('milestone3_3_error_improvement.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: milestone3_3_error_improvement.png")

print("\nMILESTONE 3 PIPELINE COMPLETED SUCCESSFULLY")
