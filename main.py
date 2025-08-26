import numpy as np
import pandas as pd
import time
import gc
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score
from imblearn.over_sampling import SMOTE
from ctgan import CTGAN


# ------------------------------------------------------------------------------------
# --- Section 1: Helper Functions for Reporting and Training ---
# ------------------------------------------------------------------------------------
def evaluate_and_report(model_name, training_time, y_true, y_pred):
    """Calculates and prints a custom performance report."""
    print(f"\n--- Performance Report for: {model_name} ---")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Fraud Precision: {precision_score(y_true, y_pred, pos_label=1, zero_division=0):.4f}")
    print(f"Fraud Recall: {recall_score(y_true, y_pred, pos_label=1, zero_division=0):.4f}")
    print(f"Fraud F1-Score: {f1_score(y_true, y_pred, pos_label=1, zero_division=0):.4f}")
    print("--------------------------------------------------")
    print("Full Classification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


def run_smote_ctgan_pipeline(X_train_df, y_train, X_test_df, categorical_features):
    """
    Encapsulates the full SMOTE + CTGAN data augmentation pipeline.
    Returns the final augmented training and encoded testing dataframes.
    """
    print("--- Running SMOTE+CTGAN Pipeline ---")

    # 1. One-hot encode training data for SMOTE
    X_train_smote_ready = pd.get_dummies(X_train_df, columns=categorical_features)

    # 2. Apply SMOTE
    print("Applying SMOTE...")
    smote = SMOTE(sampling_strategy='minority', random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_smote_ready, y_train)

    # 3. Prepare SMOTE'd data for CTGAN training
    print("Preparing data for CTGAN...")
    train_smote_df = pd.concat([X_train_smote, y_train_smote], axis=1)
    type_cols = [col for col in X_train_smote.columns if
                 col.startswith(tuple([f'{cat}_' for cat in categorical_features]))]
    fraud_data_for_gan = train_smote_df[train_smote_df['isFraud'] == 1].drop(['isFraud'], axis=1)
    # Reverse one-hot encoding for CTGAN's discrete column handling
    fraud_data_for_gan['type'] = fraud_data_for_gan[type_cols].idxmax(axis=1).str.replace('type_', '')
    fraud_data_for_gan = fraud_data_for_gan.drop(columns=type_cols)

    # 4. Train CTGAN
    print(f"Training CTGAN model on {len(fraud_data_for_gan)} SMOTE'd fraud samples...")
    ctgan = CTGAN(epochs=20, verbose=True)
    ctgan.fit(fraud_data_for_gan, discrete_columns=categorical_features)

    # 5. Generate synthetic samples
    num_legit = y_train.value_counts()[0]
    num_fraud_orig = y_train.value_counts()[1]
    num_to_generate = num_legit - num_fraud_orig
    print(f"Generating {num_to_generate} synthetic fraud samples...")
    synthetic_fraud_df = ctgan.sample(num_to_generate)

    # 6. Assemble final augmented training set
    print("Assembling final augmented dataset...")
    X_train_legit = X_train_df[y_train == 0]
    X_train_fraud_orig = X_train_df[y_train == 1]
    X_train_augmented = pd.concat([X_train_legit, X_train_fraud_orig, synthetic_fraud_df]).reset_index(drop=True)
    y_train_augmented = pd.Series(
        np.concatenate([np.zeros(len(X_train_legit)), np.ones(len(X_train_fraud_orig) + len(synthetic_fraud_df))]),
        name='isFraud'
    )

    # 7. One-hot encode the final training and testing sets
    X_train_augmented_encoded = pd.get_dummies(X_train_augmented, columns=categorical_features)
    X_test_encoded = pd.get_dummies(X_test_df, columns=categorical_features)

    # Align columns - crucial for consistent feature sets
    train_cols = X_train_augmented_encoded.columns
    test_cols = X_test_encoded.columns
    missing_in_test = set(train_cols) - set(test_cols)
    for c in missing_in_test:
        X_test_encoded[c] = 0
    missing_in_train = set(test_cols) - set(train_cols)
    for c in missing_in_train:
        X_train_augmented_encoded[c] = 0
    X_test_encoded = X_test_encoded[train_cols]

    # 8. Clean up memory
    del train_smote_df, fraud_data_for_gan, synthetic_fraud_df, X_train_legit, X_train_fraud_orig
    gc.collect()

    print("--- SMOTE+CTGAN Pipeline Complete ---")
    return X_train_augmented_encoded, y_train_augmented, X_test_encoded


# ------------------------------------------------------------------------------------
# --- Section 2: Data Loading and Preprocessing ---
# ------------------------------------------------------------------------------------
print("--- Section 2: Data Loading and Preprocessing ---")
try:
    dataset = pd.read_csv('./dataset/transactions.csv').sample(n=1000000, random_state=42)
    print(f"Dataset loaded successfully with a sample of {len(dataset)} rows.")
except FileNotFoundError:
    print("Error: 'transactions.csv' not found. Make sure it's in the './dataset/' directory.")
    exit()

dataset = dataset.dropna()
dataset.drop(['nameOrig', 'nameDest'], axis=1, inplace=True)
dataset_eng = dataset.copy()
dataset_eng['balance_diff'] = dataset_eng['newbalanceOrig'] - dataset_eng['oldbalanceOrg']
dataset_eng['balance_change'] = (dataset_eng['newbalanceOrig'] - dataset_eng['oldbalanceOrg']) / dataset_eng[
    'oldbalanceOrg'] * 100
dataset_eng.replace([np.inf, -np.inf], np.nan, inplace=True)
dataset_eng = dataset_eng.dropna()
dataset_eng['type'] = dataset_eng['type'].astype('category')
categorical_features = ['type']

# ------------------------------------------------------------------------------------
# --- Section 3: Feature Importance Analysis and Selection ---
# ------------------------------------------------------------------------------------
print("\n--- Section 3: Feature Importance Analysis and Selection ---")

X_full_eng = dataset_eng.drop(['isFraud', 'isFlaggedFraud'], axis=1)
y_full_eng = dataset_eng['isFraud']
X_full_eng_encoded = pd.get_dummies(X_full_eng, columns=categorical_features)

print("Training a preliminary model to rank feature importance...")
prelim_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
prelim_model.fit(X_full_eng_encoded, y_full_eng)

importances = prelim_model.feature_importances_
feature_names = X_full_eng_encoded.columns
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values(by='importance',
                                                                                                        ascending=False)

print("\nFeature Importance Ranking:")
print(feature_importance_df)

IMPORTANCE_THRESHOLD = 0.01
useless_features = feature_importance_df[feature_importance_df['importance'] < IMPORTANCE_THRESHOLD]['feature'].tolist()

print(f"\nIdentified {len(useless_features)} features with importance less than {IMPORTANCE_THRESHOLD}:")
print(useless_features)

base_useless_features = [f for f in useless_features if
                         not f.startswith(tuple([f'{cat}_' for cat in categorical_features]))]
dataset_selected_features = dataset_eng.drop(columns=base_useless_features)
print(f"\nCreated final dataset with {dataset_selected_features.shape[1]} columns (including target).")

# ------------------------------------------------------------------------------------
# --- Section 4: Data Splitting (For Main Experiments) ---
# ------------------------------------------------------------------------------------
print("\n--- Section 4: Data Splitting ---")
X_selected = dataset_selected_features.drop(['isFraud', 'isFlaggedFraud'], axis=1)
y_selected = dataset_selected_features['isFraud']
X_train_df, X_test_df, y_train, y_test = train_test_split(X_selected, y_selected, test_size=0.2, random_state=42,
                                                          stratify=y_selected)
print(f"Shape of main training data (selected features): {X_train_df.shape}")
print(f"Shape of main testing data (selected features): {X_test_df.shape}")

# ------------------------------------------------------------------------------------
# --- Section 5: Baseline Model (HRF-Only with SMOTE) ---
# ------------------------------------------------------------------------------------
print("\n--- Section 5: Baseline Model (HRF-Only with SMOTE) ---")
print("Applying SMOTE...")
X_train_smote_ready = pd.get_dummies(X_train_df, columns=categorical_features)
smote = SMOTE(sampling_strategy='minority', random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_smote_ready, y_train)
print(f"Shape of training data after SMOTE: {X_train_smote.shape}")

clf_smote_only = RandomForestClassifier(n_estimators=100, random_state=42)
start_time = time.time()
clf_smote_only.fit(X_train_smote, y_train_smote)
training_time_smote_only = time.time() - start_time
X_test_encoded_baseline = pd.get_dummies(X_test_df, columns=categorical_features)
# Align columns
train_cols = X_train_smote.columns
test_cols = X_test_encoded_baseline.columns
missing_in_test = set(train_cols) - set(test_cols)
for c in missing_in_test:
    X_test_encoded_baseline[c] = 0
X_test_encoded_baseline = X_test_encoded_baseline[train_cols]

y_pred_smote_only = clf_smote_only.predict(X_test_encoded_baseline)
evaluate_and_report("Baseline HRF-Only (SMOTE)", training_time_smote_only, y_test, y_pred_smote_only)

# ------------------------------------------------------------------------------------
# --- Section 6: Proposed Model (HRF-GAN on Selected Features) ---
# ------------------------------------------------------------------------------------
print("\n--- Section 6: Proposed Model (HRF-GAN) ---")
# Run the full pipeline on the main, feature-selected data
X_train_aug_main, y_train_aug_main, X_test_enc_main = run_smote_ctgan_pipeline(X_train_df, y_train, X_test_df,
                                                                               categorical_features)

clf_hrf_gan = RandomForestClassifier(n_estimators=100, random_state=42)
start_time = time.time()
clf_hrf_gan.fit(X_train_aug_main, y_train_aug_main)
training_time_hrf_gan = time.time() - start_time
y_pred_hrf_gan = clf_hrf_gan.predict(X_test_enc_main)
evaluate_and_report("Proposed HRF-GAN", training_time_hrf_gan, y_test, y_pred_hrf_gan)

# ------------------------------------------------------------------------------------
# --- Section 7: Ablation Study - Feature Engineering & Selection ---
# ------------------------------------------------------------------------------------
print("\n--- Section 7: Ablation Study - Feature Engineering & Selection ---")
print("\nThis section runs the full SMOTE+CTGAN+HRF pipeline on three different feature sets for a fair comparison.")

# --- Model A: Pipeline on data WITHOUT engineered features ---
print("\n--- Training Ablation Model A: Pipeline WITHOUT Engineered Features ---")
X_no_eng = dataset.drop(['isFraud', 'isFlaggedFraud'], axis=1)
y_no_eng = dataset['isFraud']
X_train_df_A, X_test_df_A, y_train_A, y_test_A = train_test_split(
    X_no_eng, y_no_eng, test_size=0.2, random_state=42, stratify=y_no_eng
)

X_train_aug_A, y_train_aug_A, X_test_enc_A = run_smote_ctgan_pipeline(
    X_train_df_A, y_train_A, X_test_df_A, categorical_features
)
clf_A = RandomForestClassifier(n_estimators=100, random_state=42)
start_time = time.time()
clf_A.fit(X_train_aug_A, y_train_aug_A)
training_time_A = time.time() - start_time
y_pred_A = clf_A.predict(X_test_enc_A)
evaluate_and_report("Ablation Model A (Without Eng. Features)", training_time_A, y_test_A, y_pred_A)

# --- Model B: Pipeline on data WITH ALL engineered features (before selection) ---
print("\n--- Training Ablation Model B: Pipeline WITH ALL Engineered Features ---")
X_all_eng = dataset_eng.drop(['isFraud', 'isFlaggedFraud'], axis=1)
y_all_eng = dataset_eng['isFraud']
X_train_df_B, X_test_df_B, y_train_B, y_test_B = train_test_split(
    X_all_eng, y_all_eng, test_size=0.2, random_state=42, stratify=y_all_eng
)

X_train_aug_B, y_train_aug_B, X_test_enc_B = run_smote_ctgan_pipeline(
    X_train_df_B, y_train_B, X_test_df_B, categorical_features
)
clf_B = RandomForestClassifier(n_estimators=100, random_state=42)
start_time = time.time()
clf_B.fit(X_train_aug_B, y_train_aug_B)
training_time_B = time.time() - start_time
y_pred_B = clf_B.predict(X_test_enc_B)
evaluate_and_report("Ablation Model B (With All Eng. Features)", training_time_B, y_test_B, y_pred_B)

# --- Model C: Pipeline on data WITH SELECTED engineered features (final dataset) ---
print("\n--- Training Ablation Model C: Pipeline WITH SELECTED Engineered Features ---")
X_sel_eng = dataset_selected_features.drop(['isFraud', 'isFlaggedFraud'], axis=1)
y_sel_eng = dataset_selected_features['isFraud']
X_train_df_C, X_test_df_C, y_train_C, y_test_C = train_test_split(
    X_sel_eng, y_sel_eng, test_size=0.2, random_state=42, stratify=y_sel_eng
)

X_train_aug_C, y_train_aug_C, X_test_enc_C = run_smote_ctgan_pipeline(
    X_train_df_C, y_train_C, X_test_df_C, categorical_features
)
clf_C = RandomForestClassifier(n_estimators=100, random_state=42)
start_time = time.time()
clf_C.fit(X_train_aug_C, y_train_aug_C)
training_time_C = time.time() - start_time
y_pred_C = clf_C.predict(X_test_enc_C)
evaluate_and_report("Ablation Model C (With Selected Eng. Features)", training_time_C, y_test_C, y_pred_C)

print("\nNOTE: Reports A, B, and C can now be directly compared under the same pipeline and classifier.")

# ------------------------------------------------------------------------------------
# --- Section 8: Ablation Study - Alternative Classifiers ---
# ------------------------------------------------------------------------------------
print("\n--- Section 8: Ablation Study - Alternative Classifiers ---")
subset_indices = np.random.choice(X_train_aug_main.shape[0], 50000, replace=False)
X_train_subset = X_train_aug_main.iloc[subset_indices]
y_train_subset = y_train_aug_main.iloc[subset_indices]

print("\nTraining SVM Classifier (on a subset of 50,000 samples)...")
svm_clf = SVC(random_state=42)
start_time = time.time()
svm_clf.fit(X_train_subset, y_train_subset)
training_time_svm = time.time() - start_time
y_pred_svm = svm_clf.predict(X_test_enc_main)
evaluate_and_report("SVM with GAN Data", training_time_svm, y_test, y_pred_svm)

print("\nTraining MLP Classifier...")
mlp_clf = MLPClassifier(random_state=42, max_iter=10, early_stopping=True, hidden_layer_sizes=(100, 50))
start_time = time.time()
mlp_clf.fit(X_train_aug_main, y_train_aug_main)
training_time_mlp = time.time() - start_time
y_pred_mlp = mlp_clf.predict(X_test_enc_main)
evaluate_and_report("MLP with GAN Data", training_time_mlp, y_test, y_pred_mlp)

# ------------------------------------------------------------------------------------
# --- Section 9: Sensitivity Analysis (Grid Search) ---
# ------------------------------------------------------------------------------------
print("\n--- Section 9: Sensitivity Analysis (Grid Search) ---")
subset_indices_gs = np.random.choice(X_train_smote.shape[0], 20000, replace=False)
X_train_smote_subset = X_train_smote.iloc[subset_indices_gs]
y_train_smote_subset = y_train_smote.iloc[subset_indices_gs]

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [20, None],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring='f1_weighted',
    verbose=2,
    return_train_score=True
)

print("Running Grid Search on a subset of data (20,000 samples)...")
grid_search.fit(X_train_smote_subset, y_train_smote_subset)

# Convert results into DataFrame
results_df = pd.DataFrame(grid_search.cv_results_)

# Select useful columns
results_summary = results_df[[
    'param_n_estimators', 'param_max_depth', 'param_min_samples_split',
    'mean_train_score', 'std_train_score',
    'mean_test_score', 'std_test_score',
    'rank_test_score'
]].sort_values(by='rank_test_score')

print("\nFull Grid Search Results (all parameter combos):")
print(results_summary)

# Save results for plotting later
results_summary.to_csv("grid_search_results.csv", index=False)

print("\nBest parameters found: ", grid_search.best_params_)


# ------------------------------------------------------------------------------------
# --- Section 10: Final Optimized Model Run ---
# ------------------------------------------------------------------------------------
print("\n--- Section 10: Final Model Training with Best Parameters ---")
best_params = grid_search.best_params_
# The training data is the same as in Section 6
final_model = RandomForestClassifier(random_state=42, **best_params, n_jobs=-1)
print(f"Training final HRF-GAN model on the GAN-augmented dataset with best params: {best_params}")
start_time = time.time()
final_model.fit(X_train_aug_main, y_train_aug_main)
final_training_time = time.time() - start_time
y_pred_final = final_model.predict(X_test_enc_main)
evaluate_and_report("Final Optimized HRF-GAN", final_training_time, y_test, y_pred_final)