import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def patch_retune_notebook():
    nb_path = ROOT / "colab_test_retune_only.ipynb"
    if not nb_path.exists():
        print(f"ERROR: File not found: {nb_path}")
        return

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    patched_c2 = False
    patched_c4 = False

    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            source = cell.get("source", [])
            source_str = "".join(source)

            # Patch Cell 2: Branch & Restore Files
            if "REPO_URL" in source_str and "BRANCH" in source_str:
                new_source = []
                for line in source:
                    if "BRANCH" in line and "=" in line:
                        new_source.append("BRANCH    = 'experiment-rag-upgrade'\n")
                    elif "files_to_restore = [" in line:
                        new_source.append("files_to_restore = [\n")
                        new_source.append("    (['retrieval_test.db', 'retrieval_hybrid.db', 'retrieval.db'], 'artifacts/cache/retrieval_test.db'),\n")
                        new_source.append("    (['results_partial_test.jsonl', 'results_partial.jsonl'], 'artifacts/output/results_partial_test.jsonl'),\n")
                        new_source.append("    ('R2AIStage1DATA.json', 'data/R2AIStage1DATA.json')\n")
                        new_source.append("]\n# patched_restore_line")
                    elif "patched_restore_line" in line or any(x in line for x in ["retrieval_hybrid.db", "results_partial.jsonl"]):
                        # Skip old restore lines
                        continue
                    else:
                        new_source.append(line)
                cell["source"] = new_source
                patched_c2 = True

            # Patch Cell 4: Force Peak parameters & Output name
            if "rerank_retune.py" in source_str and "HIGH_CONF" in source_str:
                new_source = []
                for line in source:
                    if "submission_reranked.zip" in line:
                        line = line.replace("submission_reranked.zip", "submission_reranked_test.zip")
                    
                    if "HIGH_CONF" in line and "=" in line:
                        line = "HIGH_CONF = 0.68\n"
                    elif "SAFE" in line and "=" in line:
                        line = "SAFE      = 0.58\n"
                    elif "MAX_ART" in line and "=" in line:
                        line = "MAX_ART   = 2\n"
                    new_source.append(line)
                cell["source"] = new_source
                patched_c4 = True

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"SUCCESS: Patched {nb_path.name} (Cell 2: {patched_c2}, Cell 4: {patched_c4})")


def patch_pipeline_notebook():
    nb_path = ROOT / "colab_hybrid_rerank.ipynb"
    if not nb_path.exists():
        print(f"ERROR: File not found: {nb_path}")
        return

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    patched_c3 = False
    patched_c5 = False
    patched_c6 = False
    patched_c7 = False
    patched_c8 = False

    # Định nghĩa nội dung mới hoàn toàn cho các Cell để tránh trùng lặp cú pháp
    c3_source = [
        "import os, sys, shutil, subprocess\n",
        "\n",
        "DRIVE_DIR = '/content/drive/MyDrive/R2AI_Artifacts'\n",
        "WORK_DIR  = '/content/sme-legal-assistant'\n",
        "REPO_URL  = 'https://github.com/Platypus27-coder/sme-legal-assistant.git'\n",
        "BRANCH    = 'experiment-rag-upgrade'\n",
        "\n",
        "if os.path.exists(WORK_DIR): \n",
        "    shutil.rmtree(WORK_DIR)\n",
        "\n",
        "print(f'⬇️ Clone branch {BRANCH} từ Github...')\n",
        "r = subprocess.run(['git', 'clone', '-b', BRANCH, REPO_URL, WORK_DIR], capture_output=True, text=True)\n",
        "if r.returncode != 0:\n",
        "    print(f'❌ Lỗi Clone: {r.stderr}')\n",
        "    raise SystemExit\n",
        "\n",
        "for d in ['output', 'cache', 'raw', 'index']:\n",
        "    os.makedirs(f'{WORK_DIR}/artifacts/{d}', exist_ok=True)\n",
        "os.makedirs(f'{WORK_DIR}/data', exist_ok=True)\n",
        "\n",
        "# Khôi phục tệp câu hỏi gốc\n",
        "if os.path.exists(f'{DRIVE_DIR}/R2AIStage1DATA.json'):\n",
        "    shutil.copy2(f'{DRIVE_DIR}/R2AIStage1DATA.json', f'{WORK_DIR}/data/R2AIStage1DATA.json')\n",
        "    print('✅ Khôi phục thành công file câu hỏi R2AIStage1DATA.json')\n",
        "\n",
        "# Khôi phục kết quả sinh một phần cũ (nếu có) để chạy tiếp tục (Checkpoint)\n",
        "if os.path.exists(f'{DRIVE_DIR}/results_partial_test.jsonl'):\n",
        "    shutil.copy2(f'{DRIVE_DIR}/results_partial_test.jsonl', f'{WORK_DIR}/artifacts/output/results_partial_test.jsonl')\n",
        "    print('✅ [Checkpoint] Khôi phục kết quả đang chạy dở: results_partial_test.jsonl')\n",
        "\n",
        "sys.path.insert(0, f'{WORK_DIR}/src')\n",
        "os.chdir(WORK_DIR)\n",
        "\n",
        "from vpl.settings import SEARCH\n",
        "print(f'\\n✅ Cấu hình hệ thống hiện tại: high_conf={SEARCH.high_conf_threshold}, max_art={SEARCH.max_articles}')\n",
        "print('✅ Cell 3 Done!')\n"
    ]

    c5_source = [
        "import os, shutil, subprocess\n",
        "DRIVE_DIR   = '/content/drive/MyDrive/R2AI_Artifacts'\n",
        "BM25_LOCAL  = 'artifacts/index/bm25/corpus.pkl'\n",
        "CHROMA_DB   = 'artifacts/index/chroma/chroma.sqlite3'\n",
        "DRIVE_TAR   = f'{DRIVE_DIR}/index_built_test.tar.gz'\n",
        "\n",
        "# Khôi phục Index RAG mới đã build từ Drive (nếu có) để tiết kiệm thời gian khi chạy lại\n",
        "if os.path.exists(DRIVE_TAR) and not os.path.exists(BM25_LOCAL):\n",
        "    print('📦 [Checkpoint] Khôi phục Index làm giàu văn cảnh mới từ Drive...')\n",
        "    os.makedirs('artifacts', exist_ok=True)\n",
        "    subprocess.run(['tar', '-xzf', DRIVE_TAR, '-C', 'artifacts'], capture_output=True)\n",
        "\n",
        "if not os.path.exists(BM25_LOCAL) or not os.path.exists(CHROMA_DB):\n",
        "    print('🔄 Bắt đầu xây dựng mới Index (BM25 + ChromaDB) với Làm giàu văn cảnh — ~45 phút...')\n",
        "    if os.path.exists('artifacts/index'):\n",
        "        shutil.rmtree('artifacts/index')\n",
        "    # Thêm check=True để quẳng lỗi đỏ ra màn hình Colab nếu tiến trình index bị sập, tránh silent fail\n",
        "    subprocess.run(['python', 'run.py', 'index', '--device', 'cuda', '--reset'], capture_output=False, check=True)\n",
        "    \n",
        "    # Sao lưu lên Drive ngay sau khi build xong để dùng cho các lần sau\n",
        "    if os.path.exists(BM25_LOCAL):\n",
        "        print('\\n☁️ Sao lưu index mới lên Drive để tái sử dụng...')\n",
        "        os.makedirs(DRIVE_DIR, exist_ok=True)\n",
        "        subprocess.run(['tar', '-czf', DRIVE_TAR, '-C', 'artifacts', 'index'], capture_output=True)\n",
        "        sz = os.path.getsize(DRIVE_TAR) / 1024 / 1024\n",
        "        print(f'☁️ index_built_test.tar.gz ({sz:.0f}MB) đã được lưu an toàn!')\n",
        "\n",
        "for p, label in [(BM25_LOCAL,'BM25'), (CHROMA_DB,'ChromaDB')]:\n",
        "    ok = os.path.exists(p)\n",
        "    sz = f'{os.path.getsize(p)/1024/1024:.0f}MB' if ok else ''\n",
        "    print(f'  {\"✅\" if ok else \"❌\"} {label} {sz}')\n",
        "print('\\n✅ Cell 5 Done!')\n"
    ]

    c6_source = [
        "# Cell 6: Hybrid Retrieve với background backup thread\n",
        "import os, shutil, sqlite3, subprocess, threading, time\n",
        "\n",
        "DRIVE_DIR = '/content/drive/MyDrive/R2AI_Artifacts'\n",
        "DB_LOCAL  = 'artifacts/cache/retrieval_test.db'\n",
        "DRIVE_DB  = f'{DRIVE_DIR}/retrieval_hybrid_test.db'\n",
        "BACKUP_INTERVAL = 300  # 5 phút sao lưu một lần\n",
        "\n",
        "# Khôi phục DB từ Drive nếu đã có sẵn tiến trình cũ (Checkpoint)\n",
        "if os.path.exists(DRIVE_DB) and not os.path.exists(DB_LOCAL):\n",
        "    shutil.copy2(DRIVE_DB, DB_LOCAL)\n",
        "    try:\n",
        "        n = sqlite3.connect(DB_LOCAL).execute('SELECT COUNT(*) FROM retrieval_cache').fetchone()[0]\n",
        "        print(f'📦 [Checkpoint] Đã khôi phục tiến trình retrieval_test.db từ Drive ({n}/2000 câu)')\n",
        "    except:\n",
        "        os.remove(DB_LOCAL)\n",
        "        print('⚠️ DB khôi phục bị lỗi, bắt đầu lại')\n",
        "\n",
        "done = 0\n",
        "if os.path.exists(DB_LOCAL):\n",
        "    try:\n",
        "        done = sqlite3.connect(DB_LOCAL).execute('SELECT COUNT(*) FROM retrieval_cache').fetchone()[0]\n",
        "    except: done = 0\n",
        "\n",
        "if done >= 2000:\n",
        "    print(f'✅ Đã retrieve đủ 2000/2000 câu — Bỏ qua!')\n",
        "else:\n",
        "    print(f'🔄 Bắt đầu chạy Hybrid Retrieve: còn {2000-done} câu cần tìm kiếm...')\n",
        "    print(f'☁️ Tự động sao lưu tiến trình lên Drive mỗi {BACKUP_INTERVAL//60} phút')\n",
        "\n",
        "    # Background backup thread\n",
        "    stop_backup = threading.Event()\n",
        "    def _backup_loop():\n",
        "        count = 0\n",
        "        while not stop_backup.is_set():\n",
        "            stop_backup.wait(BACKUP_INTERVAL)\n",
        "            if stop_backup.is_set(): break\n",
        "            if os.path.exists(DB_LOCAL):\n",
        "                try:\n",
        "                    shutil.copy2(DB_LOCAL, DRIVE_DB)\n",
        "                    n = sqlite3.connect(DB_LOCAL).execute('SELECT COUNT(*) FROM retrieval_cache').fetchone()[0]\n",
        "                    count += 1\n",
        "                    print(f'   ☁️ [Auto-backup #{count}] {n}/2000 câu đã đồng bộ -> Drive')\n",
        "                except Exception as e:\n",
        "                    print(f'   ⚠️ Đồng bộ thất bại: {e}')\n",
        "\n",
        "    backup_thread = threading.Thread(target=_backup_loop, daemon=True)\n",
        "    backup_thread.start()\n",
        "\n",
        "    try:\n",
        "        # Thêm check=True để quẳng lỗi đỏ ra màn hình Colab nếu tiến trình retrieve bị sập\n",
        "        subprocess.run(\n",
        "            ['python', 'run.py', 'retrieve',\n",
        "             '--questions', 'data/R2AIStage1DATA.json',\n",
        "             '--device', 'cuda'],\n",
        "            capture_output=False,\n",
        "            check=True\n",
        "        )\n",
        "    finally:\n",
        "        stop_backup.set()\n",
        "\n",
        "    if os.path.exists(DB_LOCAL):\n",
        "        shutil.copy2(DB_LOCAL, DRIVE_DB)\n",
        "        n = sqlite3.connect(DB_LOCAL).execute('SELECT COUNT(*) FROM retrieval_cache').fetchone()[0]\n",
        "        print(f'\\n☁️ Sao lưu lần cuối thành công: {n}/2000 câu đã lưu an toàn!')\n",
        "\n",
        "print('\\n✅ Cell 6 Done!')\n"
    ]

    c7_source = [
        "# Cell 7: Reranker + Retune\n",
        "import os, shutil, subprocess\n",
        "\n",
        "DRIVE_DIR = '/content/drive/MyDrive/R2AI_Artifacts'\n",
        "ckpt = f'{DRIVE_DIR}/results_reranked_checkpoint_test.json'\n",
        "\n",
        "# Khôi phục checkpoint nếu có sẵn tiến trình cũ (Checkpoint)\n",
        "if os.path.exists(ckpt):\n",
        "    import json\n",
        "    try:\n",
        "        done_count = len(json.loads(open(ckpt, encoding='utf-8').read()))\n",
        "        print(f'🔄 [Checkpoint] Tìm thấy checkpoint đang chạy dở: {done_count}/2000 câu -> TIẾP TỤC CHẠY!')\n",
        "    except:\n",
        "        print('⚠️ Checkpoint cũ bị lỗi, bắt đầu sinh lại từ đầu')\n",
        "else:\n",
        "    print('🆕 Chạy mới hoàn toàn (chưa có checkpoint trên Drive)')\n",
        "\n",
        "print('🎯 THAM SỐ ĐỈNH CAO: HIGH_CONF=0.68 | SAFE=0.58 | MAX_ART=2')\n",
        "print('☁️ Tự động lưu checkpoint mỗi 50 câu lên Drive')\n",
        "\n",
        "# Thêm check=True để quẳng lỗi đỏ ra màn hình Colab nếu tiến trình sinh bị sập\n",
        "subprocess.run(\n",
        "    ['python', 'rerank_retune.py',\n",
        "     '--high-conf', '0.68', '--safe', '0.58',\n",
        "     '--min-art', '0', '--max-art', '2',\n",
        "     '--device', 'cuda', '--batch-size', '64',\n",
        "     '--checkpoint-every', '50'],\n",
        "    capture_output=False,\n",
        "    check=True\n",
        ")\n",
        "\n",
        "out_zip = 'artifacts/output/submission_reranked_test.zip'\n",
        "if os.path.exists(out_zip):\n",
        "    sz = os.path.getsize(out_zip) / 1024 / 1024\n",
        "    shutil.copy2(out_zip, f'{DRIVE_DIR}/submission_reranked_test.zip')\n",
        "    print(f'\\n🏆 submission_reranked_test.zip ({sz:.1f}MB) đã được xuất thành công!')\n",
        "    print(f'☁️ Đường dẫn trên Drive: {DRIVE_DIR}/submission_reranked_test.zip')\n",
        "    print('\\n📥 HÃY TẢI FILE ZIP NÀY VỀ VÀ NỘP LÊN KAGGLE ĐỂ BỨT PHÁ ĐIỂM SỐ!')\n",
        "else:\n",
        "    print('⚠️ Chưa hoàn thành! Chạy lại cell này để tiếp tục hành trình.')\n",
        "print('\\n✅ Cell 7 Done!')\n"
    ]

    c8_source = [
        "import os, shutil, subprocess\n",
        "DRIVE_DIR = '/content/drive/MyDrive/R2AI_Artifacts'\n",
        "\n",
        "# ── Chỉnh số này ──\n",
        "HIGH_CONF = 0.68\n",
        "SAFE      = 0.58\n",
        "MAX_ART   = 2\n",
        "# ──────────────────\n",
        "\n",
        "print(f'🔧 HIGH_CONF={HIGH_CONF}, SAFE={SAFE}, MAX_ART={MAX_ART}')\n",
        "# Thêm check=True để quẳng lỗi đỏ ra màn hình Colab nếu tiến trình tune bị sập\n",
        "subprocess.run(\n",
        "    ['python', 'rerank_retune.py',\n",
        "     '--high-conf', str(HIGH_CONF), '--safe', str(SAFE),\n",
        "     '--min-art', '0', '--max-art', str(MAX_ART),\n",
        "     '--device', 'cuda', '--batch-size', '64', '--reset'],\n",
        "    capture_output=False,\n",
        "    check=True\n",
        ")\n",
        "src = 'artifacts/output/submission_reranked_test.zip'\n",
        "tag = f'{HIGH_CONF}_{MAX_ART}_{SAFE}'.replace('.', '')\n",
        "if os.path.exists(src):\n",
        "    shutil.copy2(src, f'{DRIVE_DIR}/submission_{tag}.zip')\n",
        "    print(f'✅ submission_{tag}.zip saved to Drive')\n"
    ]

    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            source = cell.get("source", [])
            source_str = "".join(source)

            # Replace Cell 3
            if "REPO_URL" in source_str and "BRANCH" in source_str:
                cell["source"] = c3_source
                patched_c3 = True

            # Replace Cell 5
            elif "BM25_LOCAL" in source_str and "CHROMA_DB" in source_str and "index_built" in source_str:
                cell["source"] = c5_source
                patched_c5 = True

            # Replace Cell 6
            elif "DB_LOCAL" in source_str and "DRIVE_DB" in source_str and "run.py" in source_str:
                cell["source"] = c6_source
                patched_c6 = True

            # Replace Cell 7
            elif "rerank_retune.py" in source_str and "checkpoint-every" in source_str:
                cell["source"] = c7_source
                patched_c7 = True

            # Replace Cell 8
            elif "HIGH_CONF =" in source_str and "SAFE      =" in source_str and "MAX_ART   =" in source_str:
                cell["source"] = c8_source
                patched_c8 = True

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"SUCCESS: Patched {nb_path.name} (Cell 3: {patched_c3}, Cell 5: {patched_c5}, Cell 6: {patched_c6}, Cell 7: {patched_c7}, Cell 8: {patched_c8})")


if __name__ == "__main__":
    patch_retune_notebook()
    patch_pipeline_notebook()
