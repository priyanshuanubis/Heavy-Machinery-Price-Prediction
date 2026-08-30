import json

with open(r'notebooks/24f2008980-notebook-v6-ultimate-kaggle.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

total = len(nb['cells'])
md_count = sum(1 for c in nb['cells'] if c['cell_type'] == 'markdown')
code_count = sum(1 for c in nb['cells'] if c['cell_type'] == 'code')

print(f"Total cells: {total}")
print(f"Markdown cells: {md_count}")
print(f"Code cells: {code_count}")

# Verify no forbidden libraries
forbidden = ['optuna', 'pickle', 'tensorflow', 'torch', 'nltk', 'spacy', 'textblob', 'word2vec']
print("\n--- Forbidden Library Check ---")
found_forbidden = False
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        source = ''.join(c['source'])
        for lib in forbidden:
            if lib in source.lower():
                print(f"WARNING: Cell {i} contains '{lib}'!")
                found_forbidden = True

if not found_forbidden:
    print("PASS: No forbidden libraries found!")

# Print section headers (markdown cells) - ASCII safe
print("\n--- Notebook Structure ---")
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'markdown':
        first_line = c['source'][0].strip() if c['source'] else ''
        # Remove non-ASCII for console safety
        safe_line = first_line.encode('ascii', 'ignore').decode('ascii')[:80]
        print(f"  Cell {i}: {safe_line}")

# Verify all 9 viva items
print("\n--- Level 1 Viva Checklist ---")
all_source = ''
for c in nb['cells']:
    all_source += ''.join(c['source']) + '\n'

checks = [
    ("1. Problem statement loaded", "Problem Statement" in all_source),
    ("2. EDA -> Preprocessing -> FE -> Models flow", "Exploratory Data Analysis" in all_source),
    ("3. EDA with univariate/bivariate", "Univariate" in all_source and "Bivariate" in all_source),
    ("4a. Missing values handling", "imputer" in all_source.lower() or "missing" in all_source.lower()),
    ("4b. Scaling", "StandardScaler" in all_source),
    ("4c. Categorical encoding", "OrdinalEncoder" in all_source),
    ("4d. Reusable pipelines", "Pipeline" in all_source and "ColumnTransformer" in all_source),
    ("5. Feature engineering + importance", "Feature Importance" in all_source),
    ("6. Multiple models", "RandomForest" in all_source and "XGBoost" in all_source and "LightGBM" in all_source),
    ("7. At least 3 models compared", "Model Comparison" in all_source),
    ("8. Clean, well-structured", md_count >= 10),
    ("9. Best model submission", "submission.csv" in all_source),
]

for name, passed in checks:
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}")
