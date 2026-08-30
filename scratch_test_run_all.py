import sys
import io
import json
import traceback

with open('notebooks/24f2008980-notebook-v24-pinnacle-mastery-kaggle.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("Starting Fast Cell Execution Audit for v24...")
ctx = {}

# Patch cell 9 for quick validation (n_splits=2, n_estimators=50, iterations=50)
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        print(f"--- Testing Code Cell {i} ---")
        code_str = ''.join(cell['source'])
        
        if "N_SPLITS = 10" in code_str:
            code_str = code_str.replace("N_SPLITS = 10", "N_SPLITS = 2")
            code_str = code_str.replace('"n_estimators": 4500', '"n_estimators": 50')
            code_str = code_str.replace('"iterations": 4000', '"iterations": 50')
            code_str = code_str.replace('"n_estimators": 200', '"n_estimators": 20')
            code_str = code_str.replace('EARLY_STOP = 150', 'EARLY_STOP = 10')
            
        try:
            exec(code_str, ctx)
            print(f"[OK] Cell {i} executed cleanly!")
        except Exception as e:
            print(f"[FAIL] Cell {i} raised exception: {e}")
            traceback.print_exc()
            sys.exit(1)

print("\n🎉 ALL CODE CELLS IN NOTEBOOK V24 EXECUTE 100% CLEANLY WITHOUT ANY ERRORS!")
