import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import math
import warnings
# --- Plotting Libraries ---
import matplotlib.pyplot as plt
import seaborn as sns

# Filter out SettingWithCopyWarning for cleaner output during transform
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
# Filter out the specific Seaborn FutureWarning about palette/hue
warnings.filterwarnings('ignore', category=FutureWarning, module='seaborn._oldcore')

# --- Seaborn Style ---
sns.set_theme(style="whitegrid")

# --- Configuration ---
CSV_FILE = 'activities_with_cities.csv' # Change to your actual file name if different
TARGET_VARIABLE = 'rating_score' # Target variable
RATINGS_COUNT_COLUMN = 'ratings_count' # Column for filtering
CITY_COLUMN = 'city'
SOURCE_COLUMN = 'source'
REGION_COLUMN = 'region'

MIN_RATINGS_COUNT_FILTER = 3 # Min ratings_count for an activity
MIN_ROWS_PER_CITY_FILTER = 3 # Min activities in a city for Approach 1 normalization
MIN_ROWS_PER_GROUP_FILTER = 3 # Min activities in a source/region group for Approach 2

# Define the feature sets
new_clarity_cols = [
    "clarity_specifies_location", "clarity_specifies_core_activity",
    "clarity_indicates_activity_format", "clarity_specifies_key_transport",
    "clarity_indicates_day_trip_origin"
]
new_value_cols = [
    "value_highlights_access_benefit", "value_specifies_guidance",
    "value_specifies_ticket_inclusion", "value_mentions_food_drink",
    "value_indicates_target_audience"
]
new_experience_cols = [
    "experience_specifies_group_size", "experience_highlights_time_benefit",
    "experience_specifies_duration", "experience_uses_hook_keywords",
    "experience_highlights_unique_access"
]
new_quality_cols = [
    "quality_free_of_errors", "quality_uses_appropriate_caps",
    "quality_free_of_contact_info", "quality_free_of_price_info",
    "quality_concise_and_scannable"
]
all_boolean_cols = (
    new_clarity_cols + new_value_cols +
    new_experience_cols + new_quality_cols
)

# --- Data Loading ---
try:
    # Try reading with default comma delimiter
    df = pd.read_csv(CSV_FILE)
    print(f"Successfully loaded '{CSV_FILE}' (assuming comma delimiter).")
except Exception as e_comma:
    print(f"Failed loading with comma delimiter: {e_comma}")
    try:
        # Try reading with semicolon delimiter
        df = pd.read_csv(CSV_FILE, delimiter=';')
        print(f"Successfully loaded '{CSV_FILE}' with semicolon delimiter.")
    except Exception as e_semi:
        print(f"Failed loading with semicolon delimiter: {e_semi}")
        print(f"Error: Could not load '{CSV_FILE}'. Please check the file path and format.")
        # Create dummy data if loading fails entirely
        print("Creating dummy data for demonstration...")
        n_rows = 600
        cities = ['Venice'] * 150 + ['Rome'] * 200 + ['Granada'] * 100 + ['Florence'] * 40 + ['Lisbon'] * 10 + ['SoloCity'] * 1 + ['DuoCity'] * 2 + [None]*5 + ['']*4
        sources = ['getyourguide'] * 250 + ['viator'] * 300 + ['klook'] * 50 # Use actual source names from output
        regions = ['europe'] * 150 + ['asia'] * 150 + ['usa'] * 200 + ['africa'] * 100 # Use actual region names from output
        np.random.shuffle(cities)
        np.random.shuffle(sources)
        np.random.shuffle(regions)
        min_len = min(len(sources), len(regions), len(cities))
        data = {
            CITY_COLUMN: cities[:min_len], SOURCE_COLUMN: sources[:min_len], REGION_COLUMN: regions[:min_len],
        }
        data[RATINGS_COUNT_COLUMN] = np.random.randint(0, 50, size=min_len)
        data[TARGET_VARIABLE] = np.clip(np.random.normal(loc=4.0, scale=0.5, size=min_len), 0, 5)
        df_temp = pd.DataFrame(data)
        for col in all_boolean_cols: data[col] = np.random.randint(0, 2, size=min_len)
        df = pd.DataFrame(data)


# --- Data Validation and Filtering ---
print(f"\nInitial data rows: {len(df)}")
required_cols_present = [TARGET_VARIABLE, RATINGS_COUNT_COLUMN, CITY_COLUMN]
# Add source/region only if they exist, otherwise add placeholder later
if SOURCE_COLUMN in df.columns: required_cols_present.append(SOURCE_COLUMN)
if REGION_COLUMN in df.columns: required_cols_present.append(REGION_COLUMN)

# Check for essential columns
for col in [TARGET_VARIABLE, RATINGS_COUNT_COLUMN, CITY_COLUMN]:
     if col not in df.columns:
          raise ValueError(f"Essential column '{col}' not found in the DataFrame.")

# Handle potentially missing source/region more robustly
if SOURCE_COLUMN not in df.columns:
    print(f"Warning: Column '{SOURCE_COLUMN}' not found. Adding placeholder 'UnknownSource'.")
    df[SOURCE_COLUMN] = 'UnknownSource'
if REGION_COLUMN not in df.columns:
    print(f"Warning: Column '{REGION_COLUMN}' not found. Adding placeholder 'UnknownRegion'.")
    df[REGION_COLUMN] = 'UnknownRegion'


df[TARGET_VARIABLE] = pd.to_numeric(df[TARGET_VARIABLE], errors='coerce')
df[RATINGS_COUNT_COLUMN] = pd.to_numeric(df[RATINGS_COUNT_COLUMN], errors='coerce')
df.dropna(subset=[TARGET_VARIABLE], inplace=True)
print(f"Rows after dropping NaN target ('{TARGET_VARIABLE}'): {len(df)}")
df.dropna(subset=[RATINGS_COUNT_COLUMN], inplace=True)
df = df[df[RATINGS_COUNT_COLUMN] >= MIN_RATINGS_COUNT_FILTER]
print(f"Rows after filtering by '{RATINGS_COUNT_COLUMN}' >= {MIN_RATINGS_COUNT_FILTER}: {len(df)}")
df.dropna(subset=[CITY_COLUMN], inplace=True)
df[CITY_COLUMN] = df[CITY_COLUMN].astype(str).str.strip()
df = df[df[CITY_COLUMN] != '']
print(f"Rows after dropping missing/empty city names: {len(df)}")
city_counts = df.groupby(CITY_COLUMN)[CITY_COLUMN].transform('size')
df = df[city_counts >= MIN_ROWS_PER_CITY_FILTER]
print(f"Rows after filtering cities with < {MIN_ROWS_PER_CITY_FILTER} rows: {len(df)}")
if df.empty: raise ValueError("No valid data remaining after filtering.")
print(f"Rows ready for processing: {len(df)}")

# --- Convert Boolean Features ---
valid_boolean_cols = []
for col in all_boolean_cols:
    if col in df.columns:
        try:
            if df[col].dtype == 'object': df[col] = df[col].astype(str).str.lower().map({'true': 1, 'false': 0, 'yes': 1, 'no': 0, '1': 1, '0': 0, 'nan': 0, '': 0})
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            valid_boolean_cols.append(col)
        except Exception as e: print(f"Warning: Could not process column '{col}'. Error: {e}. Skipping.")
    else: print(f"Warning: Column '{col}' not found. Skipping.")
if not valid_boolean_cols: raise ValueError("No valid boolean columns found.")


# --- Approach 1: Within-City Normalization (Standardization) ---
print("\n--- Approach 1: Global Model with Normalized Target ---")
print("Calculating within-city statistics for normalization...")
df_norm = df.copy()
df_norm['city_mean'] = df_norm.groupby(CITY_COLUMN)[TARGET_VARIABLE].transform('mean')
df_norm['city_std'] = df_norm.groupby(CITY_COLUMN)[TARGET_VARIABLE].transform('std')
epsilon = 1e-6
df_norm['city_std'] = df_norm['city_std'].fillna(0)
df_norm['normalized_target'] = np.where(
    df_norm['city_std'] < epsilon, 0,
    (df_norm[TARGET_VARIABLE] - df_norm['city_mean']) / (df_norm['city_std'] + epsilon)
)
print("Normalization complete.")

print("\nTraining global model on normalized target variable ('rating_score')...")
X = df_norm[valid_boolean_cols]
y = df_norm['normalized_target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_norm = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, min_samples_leaf=5)
model_norm.fit(X_train, y_train)
print("Global model training complete.")

feature_importances_norm = model_norm.feature_importances_
max_importance_norm = np.max(feature_importances_norm) if np.max(feature_importances_norm) > 0 else 1
normalized_importances_norm = (feature_importances_norm / max_importance_norm) * 100
feature_scores_norm = dict(zip(valid_boolean_cols, normalized_importances_norm))
print("\nGlobal Feature Scores (0-100, Normalized 'rating_score'):")
sorted_scores_norm = dict(sorted(feature_scores_norm.items(), key=lambda item: item[1], reverse=True))
for feature, score in sorted_scores_norm.items(): print(f"{feature}: {score:.2f}")

print("\nEvaluating global model performance (normalized scale)...")
y_pred_norm = model_norm.predict(X_test)
rmse_norm = np.sqrt(mean_squared_error(y_test, y_pred_norm))
r2_norm = r2_score(y_test, y_pred_norm)
print(f"\nModel Performance (Normalized Target):")
print(f"Root Mean Squared Error (RMSE): {rmse_norm:.3f} (in std deviations)")
print(f"R-squared (R²): {r2_norm:.3f}")

city_stats = df_norm.groupby(CITY_COLUMN).agg(mean_score=(TARGET_VARIABLE, 'mean'), std_score=(TARGET_VARIABLE, 'std')).fillna({'std_score': 0})
def predict_normalized_rating_score(feature_values, model, feature_names):
  input_vector = [feature_values.get(feature, 0) for feature in feature_names]
  input_df_ordered = pd.DataFrame([input_vector], columns=feature_names); return model.predict(input_df_ordered)[0]
def convert_norm_score_to_raw_score(predicted_norm_score, city_name, city_stats_dict):
    lookup_city_name = str(city_name).strip()
    if lookup_city_name in city_stats_dict:
        stats = city_stats_dict[lookup_city_name]; epsilon = 1e-6
        raw_score = (predicted_norm_score * (stats['std_score'] + epsilon)) + stats['mean_score']
        return np.clip(raw_score, 0, 5)
    else: print(f"Warning: Stats not found for city '{city_name}'. Cannot denormalize."); return None

# --- Plot 1: Global Feature Importance ---
print("\nPlotting Global Feature Importance (Approach 1)...")
plt.figure(figsize=(10, 8))
scores_series = pd.Series(sorted_scores_norm)
# --- FIX: Assign y to hue ---
sns.barplot(x=scores_series.values, y=scores_series.index, hue=scores_series.index, palette="viridis", orient='h', legend=False)
plt.title('Global Feature Importance (Predicting Normalized Rating Score)', fontsize=14)
plt.xlabel('Normalized Importance Score (0-100)', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()


# ==============================================================================
print("\n" + "="*70)
# ==============================================================================

# --- Approach 2: Per Source/Region Segmentation ---
print("\n--- Approach 2: Per Source/Region Segmentation ---")
df_segment = df.copy()
df_segment[SOURCE_COLUMN] = df_segment[SOURCE_COLUMN].astype(str).str.strip()
df_segment[REGION_COLUMN] = df_segment[REGION_COLUMN].astype(str).str.strip()
df_segment = df_segment[(df_segment[SOURCE_COLUMN] != '') & (df_segment[REGION_COLUMN] != '')]
print(f"\nRows before group filtering: {len(df_segment)}")
df_segment['group_key'] = df_segment[SOURCE_COLUMN] + '-' + df_segment[REGION_COLUMN]
group_counts = df_segment.groupby('group_key')['group_key'].transform('size')
df_segment = df_segment[group_counts >= MIN_ROWS_PER_GROUP_FILTER]
print(f"Rows after filtering groups with < {MIN_ROWS_PER_GROUP_FILTER} rows: {len(df_segment)}")

group_models = {}
group_scores = {}
group_performance = {}
skipped_groups = []

if df_segment.empty:
    print("\nNo data remaining after filtering for Approach 2. Skipping segmentation.")
else:
    print(f"\nStarting Per-Source/Region Model Training (Target: '{TARGET_VARIABLE}')...")
    grouped = df_segment.groupby('group_key')
    for group_name, group_df in grouped:
        print(f"\nProcessing Group: {group_name} ({len(group_df)} samples)")
        X_group = group_df[valid_boolean_cols]; y_group = group_df[TARGET_VARIABLE]
        try: X_train_group, X_test_group, y_train_group, y_test_group = train_test_split(X_group, y_group, test_size=0.2, random_state=42)
        except ValueError as e: print(f"  Warning: Could not split data. Skipping. Error: {e}"); skipped_groups.append(group_name); continue
        try:
            group_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, min_samples_leaf=5); group_model.fit(X_train_group, y_train_group); group_models[group_name] = group_model
            importances = group_model.feature_importances_; max_imp = np.max(importances) if np.max(importances) > 0 else 1; norm_imp = (importances / max_imp) * 100
            scores = dict(zip(valid_boolean_cols, norm_imp)); group_scores[group_name] = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
            y_pred_group = group_model.predict(X_test_group); y_pred_group = np.clip(y_pred_group, 0, 5)
            rmse = np.sqrt(mean_squared_error(y_test_group, y_pred_group)); r2 = r2_score(y_test_group, y_pred_group)
            group_performance[group_name] = {'RMSE': rmse, 'R2': r2, 'n_samples_test': len(y_test_group)}
            print(f"  Evaluation complete (R²: {r2:.3f}, RMSE: {rmse:.3f})")
        except Exception as e: print(f"  ERROR training/evaluating model: {e}"); skipped_groups.append(group_name)

    print("\n--- Approach 2: Per-Group Training Summary ---")
    print(f"Successfully trained models for {len(group_models)} groups.")
    print(f"Skipped {len(skipped_groups)} groups: {skipped_groups}")

    if group_performance:
        perf_df = pd.DataFrame.from_dict(group_performance, orient='index').sort_values('R2', ascending=False)
        print("\n--- Feature Scores per Group (Top 5) ---")
        for group, scores in group_scores.items():
             print(f"\n{group}:"); count=0
             for feature, score in scores.items():
                 if count < 5: print(f"  - {feature}: {score:.2f}"); count+=1
                 else: break
        print("\n--- Model Performance per Group ---")
        print(perf_df.round(3))

        # --- Plot 2: Segment Performance Comparison (R²) ---
        print("\nPlotting Segment Performance (R² - Approach 2)...")
        plt.figure(figsize=(12, 6))
        colors = ['#367BB2' if r2 > 0 else '#D65F5F' for r2 in perf_df['R2']]
        # --- FIX: Assign x to hue ---
        bars = sns.barplot(x=perf_df.index, y=perf_df['R2'], hue=perf_df.index, palette=colors, legend=False)
        plt.axhline(0, color='grey', linestyle='--', linewidth=0.8)
        plt.title('Model Performance (R²) by Source-Region Segment', fontsize=14)
        plt.xlabel('Source-Region Segment', fontsize=12)
        plt.ylabel('R-squared (R²)', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        for bar in bars.patches:
             bars.annotate(f'{bar.get_height():.3f}', (bar.get_x() + bar.get_width() / 2., bar.get_height()),
                           ha='center', va='center', size=9, xytext=(0, 5 if bar.get_height() >= 0 else -10), textcoords='offset points')
        plt.tight_layout()
        plt.show()

        # --- Plot 3: Feature Importance for Top Segments ---
        print("\nPlotting Feature Importance for Top Performing Segments (Approach 2)...")
        # --- FIX: Plot up to 6 segments (covers the 5 found previously) ---
        N_TOP_SEGMENTS = min(6, len(perf_df))
        top_segments = perf_df.head(N_TOP_SEGMENTS).index.tolist()

        if top_segments:
            # --- FIX: Adjust grid size for up to 6 plots ---
            n_cols = 3
            n_rows = math.ceil(N_TOP_SEGMENTS / n_cols) # Will be 2 rows if 4, 5, or 6 segments
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows), squeeze=False) # Adjust figsize
            axes = axes.flatten()

            for i, group_name in enumerate(top_segments):
                if group_name in group_scores:
                    ax = axes[i]
                    segment_scores = pd.Series(group_scores[group_name]).head(15) # Plot top 15 features
                    # --- FIX: Assign y to hue ---
                    sns.barplot(x=segment_scores.values, y=segment_scores.index, hue=segment_scores.index, palette="viridis", orient='h', legend=False, ax=ax)
                    ax.set_title(f'Feature Importance: {group_name}\n(R²: {perf_df.loc[group_name, "R2"]:.3f})', fontsize=12)
                    ax.set_xlabel('Normalized Importance (0-100)', fontsize=10)
                    ax.set_ylabel('')
                    ax.tick_params(axis='x', labelsize=9)
                    ax.tick_params(axis='y', labelsize=9)
                else:
                    axes[i].set_visible(False)

            for j in range(i + 1, n_rows * n_cols): # Hide unused axes
                axes[j].set_visible(False)

            fig.suptitle('Feature Importance for Top Performing Segments (Approach 2)', fontsize=16, y=1.02)
            plt.tight_layout(rect=[0, 0.03, 1, 0.98])
            plt.show()
        else:
            print("No top segments found to plot feature importance.")

    else: # if group_performance is empty
        print("No models were successfully trained for Approach 2, skipping plots.")

    def predict_rating_score_per_group(source, region, feature_values, group_models_dict, feature_names, fallback_score=None):
      cleaned_source = str(source).strip(); cleaned_region = str(region).strip()
      if not cleaned_source or not cleaned_region: return fallback_score
      group_key = cleaned_source + '-' + cleaned_region
      if group_key in group_models_dict:
          model = group_models_dict[group_key]; input_vector = [feature_values.get(feature, 0) for feature in feature_names]
          input_df_ordered = pd.DataFrame([input_vector], columns=feature_names); predicted_score = model.predict(input_df_ordered)[0]
          return np.clip(predicted_score, 0, 5)
      else: return fallback_score
    # (Example prediction code remains the same)
    if group_models:
         example_group = list(group_models.keys())[0] # Get a valid group key
         example_source, example_region = example_group.split('-', 1) # Split key back
         fallback_val_segment = df_segment[TARGET_VARIABLE].mean() if not df_segment.empty else 4.0 # Add default fallback

         pred_raw_score_segment = predict_rating_score_per_group(
             example_source, example_region, {}, group_models, valid_boolean_cols, fallback_val_segment
         )
         print(f"\nExample Prediction (Approach 2) for Group '{example_group}':")
         print(f"  Predicted Raw Rating Score (no features specified): {pred_raw_score_segment:.2f}")


print("\nProcessing finished.")