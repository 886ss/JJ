"""索引构建器：chunk 切分 → embedding → 存入 Chroma。"""
import os
import chromadb
from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore

from rag.config import (
    CHUNK_SIZE, CHUNK_OVERLAP, CHROMA_COLLECTION, INDEX_DIR,
    EMBED_PROVIDER, EMBED_MODEL,
)
from rag.loader import load_all


def _setup_embed_model():
    """根据配置选择 embedding 后端。"""
    if EMBED_PROVIDER == "openai":
        from llama_index.embeddings.openai import OpenAIEmbedding
        return OpenAIEmbedding(model=EMBED_MODEL)
    else:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        return HuggingFaceEmbedding(model_name=EMBED_MODEL)


def build_index(rebuild: bool = False):
    """构建/重建向量索引。"""
    print(f"[index] 加载文档...")
    documents = load_all()
    if not documents:
        print("[index] 无文档可索引，请先放入源码或 knowledge/docs/ 下的文档")
        return None

    # Embedding
    print(f"[index] 使用 embedding: {EMBED_PROVIDER} / {EMBED_MODEL}")
    embed_model = _setup_embed_model()
    Settings.embed_model = embed_model

    # 切分策略
    Settings.node_parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    # Chroma 客户端
    chroma_path = str(INDEX_DIR / "chroma.sqlite3")
    os.makedirs(str(INDEX_DIR), exist_ok=True)
    db = chromadb.PersistentClient(path=chroma_path)

    if rebuild:
        try:
            db.delete_collection(CHROMA_COLLECTION)
        except Exception:
            pass

    chroma_collection = db.get_or_create_collection(CHROMA_COLLECTION)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print(f"[index] 正在构建索引 ({len(documents)} 个文档)...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )
    print(f"[index] 索引构建完成")
    return index


if __name__ == "__main__":
    build_index(rebuild=True)
