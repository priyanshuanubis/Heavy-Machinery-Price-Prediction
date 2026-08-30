import json
import os
import glob

notebook_files = sorted(glob.glob("notebooks/*.ipynb"))

print(f"Found {len(notebook_files)} notebook files to audit.\n")

forbidden = ['optuna', 'pickle', 'tensorflow', 'torch', 'nltk', 'spacy', 'textblob', 'word2vec']

for path in notebook_files:
    fname = os.path.basename(path)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except Exception as e:
        print(f"[FAIL] {fname}: Invalid JSON - {e}")
        continue
        
    syntax_errs = 0
    forbidden_errs = []
    
    for i, c in enumerate(nb.get('cells', [])):
        if c.get('cell_type') == 'code':
            src = ''.join(c.get('source', []))
            try:
                compile(src, f'<{fname}_cell_{i}>', 'exec')
            except SyntaxError as e:
                syntax_errs += 1
            for lib in forbidden:
                if lib in src.lower() and lib not in forbidden_errs:
                    forbidden_errs.append(lib)
                    
    status = "OK" if syntax_errs == 0 and len(forbidden_errs) == 0 else "FAIL"
    print(f"[{status}] {fname} | Cells: {len(nb.get('cells', []))} | Syntax Errs: {syntax_errs} | Forbidden: {forbidden_errs}")
