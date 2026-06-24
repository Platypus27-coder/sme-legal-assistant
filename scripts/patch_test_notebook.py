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
        "## Cell 5 — (Bứt phá) Tạo tự động cả 5 Variant tối ưu phiên bản V5\n",
        "\n",
        "Chúng ta đã xác định được đỉnh của ngưỡng lọc nằm chính xác ở khoảng **0.67 - 0.69** (ngưỡng 0.68 đạt 0.4483, nâng lên 0.70 điểm bị tụt xuống 0.4383). Phiên bản V5 tập trung khai thác đỉnh này bằng cách mở rộng sang `max_articles=3` nhưng giữ ngưỡng siêu tin cậy để vừa ăn trọn các câu 3 luật, vừa giữ sạch các câu 1-2 luật:\n",
        "- `submission_pure_max3_t68_v5.zip` (Ngưỡng 0.68, tối đa 3 bài - Dự đoán: **0.455 - 0.465** - Ứng viên số 1!)\n",
        "- `submission_pure_max3_t67_v5.zip` (Ngưỡng 0.67, tối đa 3 bài)\n",
        "- `submission_pure_max3_t69_v5.zip` (Ngưỡng 0.69, tối đa 3 bài)\n",
        "- `submission_pure_max2_t67_v5.zip` (Ngưỡng 0.67, tối đa 2 bài)\n",
        "- `submission_pure_max2_t69_v5.zip` (Ngưỡng 0.69, tối đa 2 bài)"
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
