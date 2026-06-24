"""
VietPhapLy RAG — ChromaDB vector store.

Embedding model: mainguyen9/vietlegal-e5 (fine-tuned legal VN)
Vector DB: ChromaDB (persistent local, zero server setup)


  - Dùng ChromaDB thay Qdrant+LlamaIndex → đơn giản hơn
  - Dùng domain-specific embedding thay bge-m3 generic
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from vpl.settings import CHROMA_DIR, CHUNKS_FILE, INDEX

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _load_embedding_model(device: str = "cpu"):
    """Thử load embedding models theo thứ tự ưu tiên."""
    from sentence_transformers import SentenceTransformer

    for name in INDEX.embedding_model_candidates:
        try:
            model = SentenceTransformer(name, device=device)
            print(f"✅ Loaded embedding model: {name}")
            return model, name
        except Exception as exc:
            print(f"  ⚠ {name} failed: {str(exc)[:100]}")

    raise RuntimeError("All embedding model candidates failed.")


def _load_chunks() -> list[dict[str, Any]]:
    chunks = []
    with CHUNKS_FILE.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    chunks.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return chunks


def build(device: str = "cpu", batch_size: int | None = None, reset: bool = False) -> dict[str, Any]:
    """
    Encode tất cả chunks và lưu vào ChromaDB.

    Args:
        device: 'cpu', 'cuda', 'mps'
        batch_size: override INDEX.embedding_batch_size
        reset: xóa collection cũ trước khi build
    """
    try:
        import chromadb
    except ImportError as exc:
        raise ImportError("Install chromadb: pip install chromadb") from exc

    bs = batch_size or INDEX.embedding_batch_size
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    model, model_name = _load_embedding_model(device)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if reset:
        try:
            client.delete_collection(INDEX.chroma_collection)
            print(f"  Deleted existing collection '{INDEX.chroma_collection}'")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=INDEX.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )

    print("Loading chunks...")
    chunks = _load_chunks()
    if not chunks:
        raise ValueError("No chunks found. Run `vpl ingest` first.")

    # Skip already-indexed chunks
    existing_ids: set[str] = set()
    try:
        existing_ids = set(collection.get(include=[])["ids"])
    except Exception:
        pass

    new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]
    print(f"  {len(chunks)} total, {len(existing_ids)} existing → {len(new_chunks)} to encode")

    if not new_chunks:
        print("✅ All chunks already indexed")
        return {"indexed": len(existing_ids), "skipped": 0, "model": model_name}

    checkpoint_every = 5000
    added = 0
    for start in range(0, len(new_chunks), bs):
        batch = new_chunks[start : start + bs]
        
        # Làm giàu văn cảnh cho dense vector search
        texts = []
        for c in batch:
            meta = c.get("metadata") or {}
            doc_title = meta.get("doc_title") or ""
            article_number = meta.get("article_number") or ""
            text = c.get("text", "")
            
            enriched = text
            if doc_title:
                if article_number:
                    enriched = f"{doc_title} - Điều {article_number}: {text}"
                else:
                    enriched = f"{doc_title}: {text}"
            texts.append(enriched)
            
        model.max_seq_length = INDEX.embedding_max_length
        embeddings = model.encode(
            texts,
            batch_size=bs,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()
        collection.add(
            ids=[c["chunk_id"] for c in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[c.get("metadata") or {} for c in batch],
        )
        added += len(batch)
        if added % checkpoint_every == 0 or added == len(new_chunks):
            print(f"  indexed {added}/{len(new_chunks)}...", flush=True)
            
            # Checkpoint tự động lên Drive nếu chạy trên Colab để phòng ngừa mất kết nối giữa chừng
            drive_dir = Path("/content/drive/MyDrive/R2AI_Artifacts_Test")
            if drive_dir.exists() and added % 10000 == 0:
                print(f"  ☁️ [Auto-backup Index] Đang sao lưu checkpoint ({added} chunks) lên Drive...", flush=True)
                try:
                    import subprocess
                    drive_tar = drive_dir / "index_built_test.tar.gz"
                    subprocess.run(['tar', '-czf', str(drive_tar), '-C', str(CHROMA_DIR.parents[1]), 'index'], capture_output=True)
                    print("  ✅ Checkpoint Index đã được lưu an toàn!", flush=True)
                except Exception as e:
                    print(f"  ⚠️ Sao lưu checkpoint Index thất bại: {e}", flush=True)

    print(f"✅ ChromaDB index → {CHROMA_DIR} ({added} new chunks, model={model_name})")
    return {"indexed": added, "skipped": len(existing_ids), "model": model_name}


def get_collection(device: str = "cpu"):
    """Load ChromaDB collection + embedding model cho query time."""
    try:
        import chromadb
    except ImportError as exc:
        raise ImportError("Install chromadb: pip install chromadb") from exc

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(INDEX.chroma_collection)
    model, model_name = _load_embedding_model(device)
    return collection, model
