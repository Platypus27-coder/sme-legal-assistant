import json
import os
import shutil

with open('colab_hybrid_rerank.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        if any('Restore từ tar.gz' in line for line in source):
            # Edit cell 5
            new_source = []
            skip = False
            for line in source:
                if 'if os.path.exists(DRIVE_TAR):' in line:
                    skip = True
                    new_source.append("if os.path.exists(f'{DRIVE_INDEX}/bm25/corpus.pkl'):\n")
                    new_source.append("    print('📦 Restore BM25 từ Drive...')\n")
                    new_source.append("    os.makedirs('artifacts/index/bm25', exist_ok=True)\n")
                    new_source.append("    for f in os.listdir(f'{DRIVE_INDEX}/bm25'):\n")
                    new_source.append("        shutil.copy2(f'{DRIVE_INDEX}/bm25/{f}', f'artifacts/index/bm25/{f}')\n")
                    new_source.append("    print(f'✅ BM25 OK | ChromaDB: {\"✅\" if os.path.exists(CHROMA_DB) else \"❌ cần build mới (BGE-M3)\"}')\n")
                elif skip and "print(f'✅ BM25 OK" in line:
                    skip = False
                elif not skip:
                    new_source.append(line)
            cell['source'] = new_source
        
        elif any('Hybrid Retrieve với background backup' in line for line in source):
            # Edit cell 6
            new_source = []
            for line in source:
                if '--no-reranker' in line:
                    new_source.append(line.replace(", '--no-reranker'", ""))
                else:
                    new_source.append(line)
            cell['source'] = new_source

with open('colab_hybrid_rerank.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
