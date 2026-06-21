import json
import os

with open('colab_hybrid_rerank.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        if any('Giải nén & Setup project' in line or 'ZIP_PATH' in line for line in source):
            # Edit cell 3
            new_source = [
                "import os, sys, shutil, subprocess\n",
                "DRIVE_DIR = '/content/drive/MyDrive/R2AI_Artifacts'\n",
                "WORK_DIR  = '/content/sme-legal-assistant'\n",
                "REPO_URL  = 'https://github.com/Platypus27-coder/sme-legal-assistant.git'\n",
                "BRANCH    = 'test'\n",
                "\n",
                "if os.path.exists(WORK_DIR): shutil.rmtree(WORK_DIR)\n",
                "\n",
                "print(f'⬇️ Clone branch {BRANCH} từ Github...')\n",
                "r = subprocess.run(['git', 'clone', '-b', BRANCH, REPO_URL, WORK_DIR], capture_output=True, text=True)\n",
                "if r.returncode != 0:\n",
                "    print(f'❌ Lỗi Clone (Có thể do repo private cần Token): {r.stderr}')\n",
                "    print('👉 Cách sửa: Thêm token vào URL -> https://<YOUR_TOKEN>@github.com/...')\n",
                "    raise SystemExit\n",
                "\n",
                "key_files = ['run.py', 'src/vpl/settings.py', 'rerank_retune.py', 'src/vpl/search/hybrid.py']\n",
                "print('📋 Kiểm tra cấu trúc:')\n",
                "for f in key_files:\n",
                "    ok = os.path.exists(f'{WORK_DIR}/{f}')\n",
                "    print(f'  {\"✅\" if ok else \"❌\"} {f}')\n",
                "\n",
                "for d in ['output', 'cache', 'raw', 'index']:\n",
                "    os.makedirs(f'{WORK_DIR}/artifacts/{d}', exist_ok=True)\n",
                "os.makedirs(f'{WORK_DIR}/data', exist_ok=True)\n",
                "\n",
                "shutil.copy2(f'{DRIVE_DIR}/results_partial.jsonl', f'{WORK_DIR}/artifacts/output/results_partial.jsonl')\n",
                "shutil.copy2(f'{DRIVE_DIR}/R2AIStage1DATA.json',   f'{WORK_DIR}/data/R2AIStage1DATA.json')\n",
                "\n",
                "sys.path.insert(0, f'{WORK_DIR}/src')\n",
                "os.chdir(WORK_DIR)\n",
                "\n",
                "from vpl.settings import SEARCH\n",
                "print(f'\\n✅ Import OK — high_conf={SEARCH.high_conf_threshold}, max_art={SEARCH.max_articles}')\n",
                "print('✅ Cell 3 Done!')\n"
            ]
            cell['source'] = new_source
            break

with open('colab_hybrid_rerank.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
