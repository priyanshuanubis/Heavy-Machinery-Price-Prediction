"""
Create an offline version of the notebook that:
1. Loads data from the local data/ folder
2. Prints results instead of saving submission.csv
"""
import json
import copy

# Read the existing notebook
with open(r'notebooks/24f2008980-notebook-2026t2 (3).ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Make a deep copy
nb_offline = copy.deepcopy(nb)

# --- Fix Cell: Data Loading (Cell index 4, the code cell after "Data Loading" markdown) ---
# Find the data loading code cell
for i, cell in enumerate(nb_offline['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'DATA_PATH' in source and 'train.csv' in source and 'test.csv' in source:
            # Replace data loading cell
            cell['source'] = [line + "\n" for line in """# Load data from local data/ folder
DATA_PATH = os.path.join(os.path.dirname(os.getcwd()), "data")

# Fallback: check common locations
if not os.path.isdir(DATA_PATH):
    DATA_PATH = "data"
if not os.path.isdir(DATA_PATH):
    DATA_PATH = r"e:\\HeavyEquipmentML\\data"
if os.path.isdir("/kaggle/input"):
    for root, _, files in os.walk("/kaggle/input"):
        if "train.csv" in files and "test.csv" in files:
            DATA_PATH = root
            break

print(f"Data directory: {DATA_PATH}")

# Load datasets
train_raw = pd.read_csv(os.path.join(DATA_PATH, "train.csv"), low_memory=False)
test_raw = pd.read_csv(os.path.join(DATA_PATH, "test.csv"), low_memory=False)
sample_sub = pd.read_csv(os.path.join(DATA_PATH, "sample_submission.csv"))

print(f"\\nTrain Dataset : {train_raw.shape[0]:,} rows x {train_raw.shape[1]} columns")
print(f"Test Dataset  : {test_raw.shape[0]:,} rows x {test_raw.shape[1]} columns")
print(f"Submission    : {sample_sub.shape[0]:,} rows x {sample_sub.shape[1]} columns")""".split("\n")]
            cell['outputs'] = []
            cell['execution_count'] = None
            print(f"  Updated data loading cell (index {i})")
            break

# --- Fix Cell: Final Submission — print instead of saving ---
# Find the submission code cell (contains "submission.csv" and "to_csv")
for i in range(len(nb_offline['cells']) - 1, -1, -1):
    cell = nb_offline['cells'][i]
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'submission.to_csv' in source or ('submission.csv' in source and 'to_csv' in source):
            cell['source'] = [line + "\n" for line in """# Train final model with best hyperparameters on full training data
print("Training final model on full training data...")
final_model = search.best_estimator_
final_model.fit(X, y)

# Generate predictions on test set
test_predictions_log = final_model.predict(X_test)
test_predictions = np.maximum(0, np.expm1(test_predictions_log))  # Convert back from log scale

# Create submission DataFrame (for display, not saving)
target_col = [c for c in sample_sub.columns if c != "TransactionID"][0]
submission = sample_sub.copy()
submission[target_col] = test_predictions

# --- OFFLINE MODE: Print results instead of saving ---
print("\\n" + "=" * 70)
print("PREDICTION RESULTS (Offline Mode)")
print("=" * 70)

print(f"\\nSubmission shape: {submission.shape}")
print(f"\\nPrediction Statistics:")
print(f"  Mean:   {test_predictions.mean():.2f}")
print(f"  Median: {np.median(test_predictions):.2f}")
print(f"  Std:    {test_predictions.std():.2f}")
print(f"  Min:    {test_predictions.min():.2f}")
print(f"  Max:    {test_predictions.max():.2f}")

print(f"\\nFirst 20 predictions:")
print(submission.head(20).to_string(index=False))

print(f"\\nLast 10 predictions:")
print(submission.tail(10).to_string(index=False))

# Validation checks
assert len(submission) == len(sample_sub), "Row count mismatch!"
assert submission[target_col].notna().all(), "NaN values in predictions!"
assert np.isfinite(submission[target_col]).all(), "Infinite values in predictions!"

print("\\nAll validation checks passed.")""".split("\n")]
            cell['outputs'] = []
            cell['execution_count'] = None
            print(f"  Updated submission cell (index {i})")
            break

# --- Also update the distribution plot cell right after submission ---
# Find the plot cell that references test_predictions distribution
for i in range(len(nb_offline['cells']) - 1, -1, -1):
    cell = nb_offline['cells'][i]
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'Predicted Price Distribution' in source and 'Train Actual vs Test Predicted' in source:
            cell['source'] = [line + "\n" for line in """# Distribution of predicted vs actual prices
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Predicted price distribution
sns.histplot(test_predictions, kde=True, ax=axes[0], color="coral", bins=50)
axes[0].set_title("Predicted Price Distribution (Test Set)", fontsize=13)
axes[0].set_xlabel("Predicted TargetValue")

# Training vs Predicted comparison
sns.histplot(train_raw["TargetValue"], kde=True, ax=axes[1], color="steelblue", bins=50, label="Train (Actual)", alpha=0.6)
sns.histplot(test_predictions, kde=True, ax=axes[1], color="coral", bins=50, label="Test (Predicted)", alpha=0.6)
axes[1].set_title("Train Actual vs Test Predicted Distribution", fontsize=13)
axes[1].set_xlabel("TargetValue")
axes[1].legend()

plt.tight_layout()
plt.show()

print("\\nThe predicted distribution should broadly follow the training distribution,")
print("confirming the model generalizes well to unseen data.")""".split("\n")]
            cell['outputs'] = []
            cell['execution_count'] = None
            print(f"  Updated distribution plot cell (index {i})")
            break

# --- Update the title cell to indicate offline version ---
for i, cell in enumerate(nb_offline['cells']):
    if cell['cell_type'] == 'markdown':
        source = ''.join(cell['source'])
        if 'Heavy Equipment Selling Price Prediction' in source:
            # Add "(Offline Version)" to the title
            new_source = source.replace(
                'Heavy Equipment Selling Price Prediction',
                'Heavy Equipment Selling Price Prediction (Offline Version)'
            )
            nb_offline['cells'][i]['source'] = [line + "\n" for line in new_source.split("\n")]
            print(f"  Updated title cell (index {i})")
            break

# Clear all outputs and execution counts
for cell in nb_offline['cells']:
    if cell['cell_type'] == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None

# Save the offline version
output_path = r'notebooks/24f2008980-notebook-2026t2-offline.ipynb'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(nb_offline, f, indent=1, ensure_ascii=False)

print(f"\nNotebook saved to: {output_path}")
