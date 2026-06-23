import json
from pathlib import Path

notebook_path = Path(__file__).resolve().parents[1] / "colab_test_retune_only.ipynb"

if not notebook_path.exists():
    print(f"❌ File not found: {notebook_path}")
    exit(1)

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Patch Cell 2 (directory deletion bug)
for cell in nb["cells"]:
    if cell.get("cell_type") == "code" and "shutil.rmtree(WORK_DIR)" in "".join(cell.get("source", [])):
        source = cell["source"]
        new_source = []
        for line in source:
            if "shutil.rmtree(WORK_DIR)" in line:
                new_source.append("try:\n")
                new_source.append("    os.chdir('/content')\n")
                new_source.append("except Exception: pass\n")
                new_source.append("if os.path.exists(WORK_DIR): shutil.rmtree(WORK_DIR)\n")
            else:
                new_source.append(line)
        cell["source"] = new_source
        print("SUCCESS: Patched Cell 2 in notebook")

# Patch Cell 5 (check=True and real-time logs)
nb["cells"] = [c for c in nb["cells"] if "tools/optimize_submission.py" not in "".join(c.get("source", [])) and "Cell 5" not in "".join(c.get("source", []))]

markdown_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Cell 5 — (Bứt phá) Tạo tự động cả 5 Variant tối ưu (Dựa trên Reranker thuần khiết)\n",
        "\n",
        "Script này đã loại bỏ Lexical Boost (được chứng minh làm giảm điểm xuống 0.406) để chạy Reranker thuần túy giống như file đạt 0.4136 của bạn. Nó sẽ sinh ra **5 file nộp bài** lên Drive:\n",
        "- `submission_pure_base_v2.zip` (Ngưỡng 0.62, tương tự kết quả 0.4136 của bạn)\n",
        "- `submission_pure_high_v2.zip` (Ngưỡng 0.65 - Kiểm tra xem tăng Precision có tăng điểm không)\n",
        "- `submission_pure_low_v2.zip` (Ngưỡng 0.58 - Kiểm tra xem tăng Recall có tăng điểm không)\n",
        "- `submission_pure_force_v2.zip` (Ngưỡng 0.62, bắt buộc nộp ít nhất 1 bài để không bị phạt mảng rỗng)\n",
        "- `submission_pure_max2_v2.zip` (Ngưỡng 0.62, giới hạn tối đa 2 bài - Tốt nhất nếu đề đa số là 1-2 bài đúng)"
    ]
}

code_cell = {
    "cell_type": "code",
    "metadata": {
        "id": "cell_5"
    },
    "source": [
        "import subprocess\n",
        "import torch\n",
        "\n",
        "has_gpu = torch.cuda.is_available()\n",
        "print('🔍 Kiểm tra thiết bị phần cứng:')\n",
        "print(f'   - CUDA Available: {has_gpu}')\n",
        "if has_gpu:\n",
        "    print(f'   - GPU Device: {torch.cuda.get_device_name(0)}')\n",
        "else:\n",
        "    print('   ⚠️ WARNING: Không tìm thấy GPU! Việc chạy Reranker trên CPU sẽ cực kỳ lâu (mất khoảng 2 tiếng).')\n",
        "    print('   💡 Khuyên nghị: Hãy bấm Dừng cell này, vào \"Chỉnh sửa\" -> \"Cài đặt sổ ghi chép\" -> Chọn GPU T4 rồi chạy lại.')\n",
        "\n",
        "print('\\n🚀 Bắt đầu tạo 5 Variant tối ưu nộp bài (tiến độ sẽ được hiển thị trực tiếp)...')\n",
        "cmd = [\n",
        "    'python', '-u', 'tools/optimize_submission.py',\n",
        "    '--use-reranker',\n",
        "    '--device', 'cuda' if has_gpu else 'cpu',\n",
        "    '--batch-size', '64'\n",
        "]\n",
        "\n",
        "try:\n",
        "    # Chạy python với cờ -u (unbuffered) để hiển thị log real-time\n",
        "    subprocess.run(cmd, check=True)\n",
        "    print('\\n🎉 HOÀN TẤT! Cả 5 file submission đã có trên Drive R2AI_Artifacts của bạn.')\n",
        "except subprocess.CalledProcessError as e:\n",
        "    print(f'\\n❌ Lỗi thực thi optimize_submission.py: {e}')\n"
    ],
    "execution_count": None,
    "outputs": []
}

nb["cells"].append(markdown_cell)
nb["cells"].append(code_cell)

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("SUCCESS: Patched Cell 2 and Cell 5 with 5 Variants in colab_test_retune_only.ipynb")
