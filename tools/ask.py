"""
Công cụ Hỏi - Đáp tương tác (Interactive QA Tool)
Cho phép test trực tiếp 1 câu hỏi bất kỳ bằng hệ thống RAG hiện tại.
Chạy ở Local: Sẽ tự động dùng CPU nếu không có CUDA (mặc dù LLM sẽ chạy hơi chậm).
"""

import argparse
import sys

from vpl.search.expander import expand
from vpl.search.hybrid import HybridRetriever
from vpl.search.reranker import load_reranker
from vpl.store.vectors import get_collection
from vpl.answer.generator import LegalGenerator
from vpl.answer.postprocess import PostConfig, PostProcessor

def main():
    parser = argparse.ArgumentParser(description="Test 1 câu hỏi pháp lý với hệ thống RAG.")
    parser.add_argument("question", type=str, help="Câu hỏi pháp lý cần giải đáp")
    parser.add_argument("--device", type=str, default="cuda", help="Thiết bị chạy (cuda/cpu)")
    args = parser.parse_args()

    question = args.question
    device = args.device

    print(f"\n[1/3] Đang phân tích câu hỏi và mở rộng (HyDE)...")
    expanded = expand(question)
    print(f"      HyDE: {expanded}")

    print(f"\n[2/3] Đang tìm kiếm tài liệu liên quan (Retrieval & Reranking)...")
    try:
        collection, embed_model = get_collection(device=device)
        reranker = load_reranker(device=device)
        retriever = HybridRetriever(
            chroma_collection=collection,
            embedding_model=embed_model,
            reranker=reranker,
            device=device,
        )
        chunks = retriever.retrieve(question, expanded_query=expanded)
    except Exception as e:
        print(f"❌ Lỗi khi tìm kiếm: {e}")
        sys.exit(1)

    print(f"\n[3/3] Đang sinh câu trả lời (Generation) bằng LLM...")
    try:
        # Load LLM
        generator = LegalGenerator.from_pretrained()
        postprocessor = PostProcessor(PostConfig())

        # Chọn chunks theo threshold
        selected_chunks = postprocessor.select_relevant_chunks(chunks)
        
        if not selected_chunks:
            print("⚠️ Không tìm thấy điều luật nào thỏa mãn ngưỡng an toàn (SAFE_CONF).")
        else:
            print("\n📚 Các tài liệu hệ thống dùng làm ngữ cảnh:")
            for i, c in enumerate(selected_chunks, 1):
                meta = c.metadata if hasattr(c, "metadata") else c.get("metadata", {})
                print(f"   {i}. {meta.get('formatted_article')} (Điểm: {c.score:.4f})")

        # Sinh câu trả lời thô
        raw_answers = generator.generate([question], [selected_chunks])
        raw_answer = raw_answers[0]

        # Hậu xử lý (Lọc ảo giác, chèn trích dẫn)
        final_answer, hallucinated = postprocessor.process_answer(raw_answer, selected_chunks)

        print("\n" + "="*60)
        print("💡 CÂU TRẢ LỜI CỦA HỆ THỐNG:")
        print("="*60)
        print(final_answer)
        print("="*60)
        
        if hallucinated:
            print(f"\n🚨 [Anti-Hallucination] Đã tự động xóa bỏ các điều luật ảo do LLM bịa ra: {hallucinated}")

    except Exception as e:
        print(f"❌ Lỗi khi sinh câu trả lời: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
