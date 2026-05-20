import os
from pathlib import Path
from llama_index.core import SimpleDirectoryReader, Document

from rag.config import PROJECT_DIR, SOURCE_DIRS, INCLUDE_FILES, EXCLUDE_DIRS, DOCS_DIR


def _iter_source_files() -> list[Path]:
    """收集所有源码和文档文件，排除无关目录。"""
    files: list[Path] = []
    for src_dir_name in SOURCE_DIRS:
        src_dir = PROJECT_DIR / src_dir_name
        if not src_dir.exists():
            continue
        for root, dirs, filenames in os.walk(src_dir):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            for fname in filenames:
                fpath = Path(root) / fname
                if any(fpath.match(p) for p in INCLUDE_FILES):
                    files.append(fpath)
    return files


def load_source_files() -> list[Document]:
    """加载项目源码文件为 LlamaIndex Document 列表。"""
    files = _iter_source_files()
    reader = SimpleDirectoryReader(input_files=[str(f) for f in files])
    return reader.load_data(show_progress=True)


def load_docs() -> list[Document]:
    """加载 knowledge/docs/ 下的文档。"""
    if not DOCS_DIR.exists() or not any(DOCS_DIR.iterdir()):
        return []
    reader = SimpleDirectoryReader(input_dir=str(DOCS_DIR))
    return reader.load_data(show_progress=True)


def load_all() -> list[Document]:
    """加载全部文档：源码 + docs 目录。"""
    docs = load_source_files()
    docs += load_docs()
    print(f"[loader] 共加载 {len(docs)} 个文档")
    return docs


if __name__ == "__main__":
    docs = load_all()
    for d in docs[:5]:
        src = d.metadata.get("file_path", "?")
        print(f"  [{src}] {d.text[:120]}...")
