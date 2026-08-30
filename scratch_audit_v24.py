import json
import os

def audit_notebook(path):
    print("=" * 60)
    print(f"AUDITING: {path}")
    print("=" * 60)
    
    if not os.path.exists(path):
        print(f"ERROR: File not found: {path}")
        return False
        
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    total_cells = len(nb['cells'])
    md_cells = [c for c in nb['cells'] if c['cell_type'] == 'markdown']
    code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
    
    print(f"Total Cells: {total_cells} (Markdown: {len(md_cells)}, Code: {len(code_cells)})")
    
    # 1. Syntax Check
    syntax_errors = 0
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            src = ''.join(cell['source'])
            try:
                compile(src, f'<cell_{i}>', 'exec')
            except SyntaxError as e:
                print(f"[FAIL] Syntax Error in Code Cell {i}: {e}")
                syntax_errors += 1
                
    if syntax_errors == 0:
        print("[PASS] Python Syntax Check: OK")
    else:
        print(f"[FAIL] Python Syntax Check: {syntax_errors} errors found!")
        
    # 2. Forbidden Libraries
    forbidden = ['optuna', 'pickle', 'tensorflow', 'torch', 'nltk', 'spacy', 'textblob', 'word2vec']
    forbidden_found = False
    all_code_src = '\n'.join([''.join(c['source']) for c in code_cells])
    for lib in forbidden:
        if lib in all_code_src.lower():
            print(f"[FAIL] Forbidden Library Detected: '{lib}'")
            forbidden_found = True
    if not forbidden_found:
        print("[PASS] Forbidden Libraries Check: OK")
        
    # 3. Viva Checklist
    all_src = '\n'.join([''.join(c['source']) for c in nb['cells']])
    viva_checks = [
        ("Problem Statement & Objective", "Problem Statement" in all_src),
        ("Exploratory Data Analysis (EDA)", "Exploratory Data Analysis" in all_src or "EDA" in all_src),
        ("Univariate Analysis", "histplot" in all_src or "Univariate" in all_src),
        ("Bivariate Analysis / Correlation", "heatmap" in all_src or "Bivariate" in all_src or "scatter" in all_src or "barplot" in all_src),
        ("Missing Value Handling", "isnull" in all_src or "isna" in all_src or "fillna" in all_src or "imputer" in all_src),
        ("Categorical Encoding", "OrdinalEncoder" in all_src or "Target" in all_src or "category" in all_src),
        ("Feature Engineering", "engineer_features" in all_src or "Feature" in all_src),
        ("Feature Importance Analysis", "feature_importances_" in all_src or "Feature Importance" in all_src),
        ("Multiple Models (>= 3)", "LGBMRegressor" in all_src and "CatBoostRegressor" in all_src and "XGBRegressor" in all_src),
        ("Model Comparison Table/Plot", "results_df" in all_src or "Comparison" in all_src),
        ("Level-2 Meta Ensemble / Stacking", "RidgeCV" in all_src or "minimize" in all_src),
        ("Submission CSV Export", "submission.csv" in all_src),
    ]
    
    print("\n--- Viva & Assignment Compliance Checklist ---")
    all_viva_passed = True
    for item, passed in viva_checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_viva_passed = False
        print(f"  [{status}] {item}")
        
    # 4. LaTeX Escaping Check
    latex_issues = 0
    for i, c in enumerate(md_cells):
        src = ''.join(c['source'])
        if "\\\\\\\\" in src or "\\\\log" in src or "\\\\text" in src or "\\\\d" in src:
            print(f"  [WARN] LaTeX Double Escaping issue in MD Cell {i}: backslashes '\\\\'")
            latex_issues += 1
            
    if latex_issues == 0:
        print("[PASS] Markdown LaTeX Formatting Check: OK")
        
    return syntax_errors == 0 and not forbidden_found and all_viva_passed

audit_notebook("notebooks/24f2008980-notebook-v24-pinnacle-mastery-kaggle.ipynb")
