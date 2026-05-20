"""查询引擎：基于已构建的索引进行 RAG 查询。"""
import os
from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.core.llms import LLMMetadata
from llama_index.core.types import MessageRole
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI
import chromadb

from rag.config import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, LLM_MODEL,
    EMBED_PROVIDER, EMBED_MODEL,
    CHROMA_COLLECTION, INDEX_DIR,
)
from rag.index_builder import _setup_embed_model, build_index


class DeepSeekLLM(OpenAI):
    """OpenAI 兼容 LLM 适配器，绕过模型名校验以支持 DeepSeek。"""

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=65536,
            num_output=self.max_tokens or 4096,
            is_chat_model=True,
            is_function_calling_model=True,
            model_name=self.model,
            system_role=MessageRole.SYSTEM,
        )


def _setup_llm() -> DeepSeekLLM:
    return DeepSeekLLM(
        model=LLM_MODEL,
        api_key=DEEPSEEK_API_KEY,
        api_base=DEEPSEEK_BASE_URL,
    )


def _load_index():
    """加载已有索引，若无则自动构建。"""
    chroma_path = str(INDEX_DIR / "chroma.sqlite3")
    if not os.path.exists(chroma_path):
        print("[query] 索引不存在，自动构建...")
        return build_index(rebuild=False)

    Settings.embed_model = _setup_embed_model()
    Settings.llm = _setup_llm()

    db = chromadb.PersistentClient(path=chroma_path)
    try:
        chroma_collection = db.get_collection(CHROMA_COLLECTION)
    except Exception:
        print("[query] 集合不存在，自动构建...")
        return build_index(rebuild=False)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store)
    return index


def query(question: str, rebuild: bool = False) -> str:
    """向 RAG 知识库提问。"""
    if not DEEPSEEK_API_KEY:
        return "[query] 错误：请先在 .env 中设置 DEEPSEEK_API_KEY"

    Settings.embed_model = _setup_embed_model()
    Settings.llm = _setup_llm()

    if rebuild:
        index = build_index(rebuild=True)
    else:
        index = _load_index()

    query_engine = index.as_query_engine(similarity_top_k=5)
    response = query_engine.query(question)
    return str(response)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python -m rag.query_engine \"你的问题\"")
        sys.exit(1)
    ans = query(sys.argv[1])
    print(f"\n回答:\n{ans}")
