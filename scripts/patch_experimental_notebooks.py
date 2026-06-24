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

    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            source = cell.get("source", [])
            source_str = "".join(source)

            # Patch Cell 3: Branch & Restore Files
            if "REPO_URL" in source_str and "BRANCH" in source_str:
                new_source = []
                for line in source:
                    if "BRANCH" in line and "=" in line:
                        new_source.append("BRANCH    = 'experiment-rag-upgrade'\n")
                    elif "results_partial.jsonl" in line and "shutil.copy2" in line:
                        new_source.append("if os.path.exists(f'{DRIVE_DIR}/results_partial_test.jsonl'):\n")
                        new_source.append("    shutil.copy2(f'{DRIVE_DIR}/results_partial_test.jsonl', f'{WORK_DIR}/artifacts/output/results_partial_test.jsonl')\n")
                        new_source.append("    print('[OK] Restored results_partial_test.jsonl')\n")
                        new_source.append("elif os.path.exists(f'{DRIVE_DIR}/results_partial.jsonl'):\n")
                        new_source.append("    shutil.copy2(f'{DRIVE_DIR}/results_partial.jsonl', f'{WORK_DIR}/artifacts/output/results_partial_test.jsonl')\n")
                        new_source.append("    print('[OK] Restored results_partial.jsonl as results_partial_test.jsonl')\n")
                    else:
                        new_source.append(line)
                cell["source"] = new_source
                patched_c3 = True

            # Patch Cell 5: Force Rebuild the Index to apply Contextual Enrichment
            if "BM25_LOCAL" in source_str and "CHROMA_DB" in source_str and "index_built.tar.gz" in source_str:
                new_source = []
                # Check if already patched to avoid duplicating FORCE_REBUILD
                if "FORCE_REBUILD" not in source_str:
                    for line in source:
                        if "if os.path.exists(f'{DRIVE_INDEX}/bm25/corpus.pkl'):" in line:
                            new_source.append("FORCE_REBUILD = True  # Bắt buộc build lại để cập nhật Contextual Enrichment đột phá mới!\n\n")
                            new_source.append("if FORCE_REBUILD:\n")
                            new_source.append("    print('⚠️ FORCE_REBUILD is True. Bỏ qua restore từ Drive, tiến hành xóa index cũ để build mới...')\n")
                            new_source.append("    if os.path.exists('artifacts/index'):\n")
                            new_source.append("        shutil.rmtree('artifacts/index')\n")
                            new_source.append("else:\n")
                            new_source.append("    if os.path.exists(f'{DRIVE_INDEX}/bm25/corpus.pkl'):\n")
                        elif "if os.path.exists(f'{DRIVE_INDEX}/bm25/corpus.pkl'):" not in line and any(x in line for x in [
                            "print('📦 Restore BM25 từ Drive...')",
                            "os.makedirs('artifacts/index/bm25', exist_ok=True)",
                            "shutil.copy2(f'{DRIVE_INDEX}/bm25/{f}', f'artifacts/index/bm25/{f}')",
                            "print(f'✅ BM25 OK | ChromaDB: {\"✅\" if os.path.exists(CHROMA_DB) else \"❌ cần build mới (BGE-M3)\"}')"
                        ]):
                            # Indent existing restore lines inside the else block
                            new_source.append("    " + line)
                        else:
                            new_source.append(line)
                    cell["source"] = new_source
                    patched_c5 = True
                else:
                    patched_c5 = True

            # Patch Cell 6: Database paths for retrieval
            if "DB_LOCAL" in source_str and "DRIVE_DB" in source_str and "run.py" in source_str:
                new_source = []
                for line in source:
                    if "DB_LOCAL" in line and "retrieval.db" in line:
                        new_source.append("DB_LOCAL  = 'artifacts/cache/retrieval_test.db'\n")
                    elif "DRIVE_DB" in line and "retrieval_hybrid.db" in line:
                        new_source.append("DRIVE_DB  = f'{DRIVE_DIR}/retrieval_hybrid_test.db'\n")
                    elif "SELECT COUNT(*)" in line:
                        new_source.append(line.replace("retrieval_cache", "retrieval_cache"))
                    else:
                        new_source.append(line)
                cell["source"] = new_source
                patched_c6 = True

            # Patch Cell 7: Checkpoint path & Force Peak parameters (0.68/0.58/max2)
            if "rerank_retune.py" in source_str and "checkpoint-every" in source_str:
                new_source = []
                for line in source:
                    # Replace filenames with test suffix if not already done
                    line = line.replace("results_reranked_checkpoint.json", "results_reranked_checkpoint_test.json")
                    line = line.replace("submission_reranked.zip", "submission_reranked_test.zip")
                    
                    # Force Peak parameters in command args
                    if "'--high-conf'" in line:
                        line = "     '--high-conf', '0.68',\n"
                    elif "'--safe'" in line:
                        line = "     '--safe', '0.58',\n"
                    elif "'--max-art'" in line:
                        line = "     '--max-art', '2',\n"
                    
                    # Update print statements to show correct settings
                    if "HIGH_CONF=" in line:
                        line = "    print('   HIGH_CONF=0.68 | SAFE=0.58 | MAX_ART=2')\n"
                    
                    new_source.append(line)
                cell["source"] = new_source
                patched_c7 = True

            # Patch Cell 8 (Optional retune cell): Force Peak parameters
            if "HIGH_CONF =" in source_str and "SAFE      =" in source_str and "MAX_ART   =" in source_str:
                new_source = []
                for line in source:
                    if "HIGH_CONF =" in line:
                        line = "HIGH_CONF = 0.68\n"
                    elif "SAFE      =" in line:
                        line = "SAFE      = 0.58\n"
                    elif "MAX_ART   =" in line:
                        line = "MAX_ART   = 2\n"
                    new_source.append(line)
                cell["source"] = new_source
                patched_c8 = True

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"SUCCESS: Patched {nb_path.name} (Cell 3: {patched_c3}, Cell 5: {patched_c5}, Cell 6: {patched_c6}, Cell 7: {patched_c7}, Cell 8: {patched_c8})")


if __name__ == "__main__":
    patch_retune_notebook()
    patch_pipeline_notebook()
