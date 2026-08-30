import json

nb = json.load(open('notebooks/24f2008980-notebook-v22-milestone-mastery-kaggle.ipynb'))
issues = 0
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        full_src = ''.join(cell['source'])
        try:
            compile(full_src, f'<cell_{i}>', 'exec')
        except SyntaxError as e:
            print(f"SYNTAX ERROR in cell {i}: {e}")
            issues += 1

if issues == 0:
    print("OK: All code cells in v22 pass Python syntax check!")
else:
    print(f"FAIL: {issues} cells have syntax errors!")
