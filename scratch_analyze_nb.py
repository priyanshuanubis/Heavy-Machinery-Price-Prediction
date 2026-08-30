import json

with open(r'notebooks/24f2008980-notebook-2026t2 (3).ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    cell_type = c['cell_type']
    source = ''.join(c['source'])
    # Print first 200 chars of source
    preview = source[:300].replace('\n', '\\n')
    print(f"Cell {i} [{cell_type}]: {preview}")
    print("---")
