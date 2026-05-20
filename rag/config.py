import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- API 配置 ---
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

# --- Embedding 配置 ---
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "huggingface")  # huggingface | openai
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-zh-v1.5")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# --- 路径 ---
PROJECT_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_DIR / "knowledge" / "docs"
INDEX_DIR = PROJECT_DIR / "knowledge" / "indexes"

# --- 文档加载配置 ---
SOURCE_DIRS = ["ds-portfolio"]
INCLUDE_FILES = ["*.py", "*.md", "*.txt", "*.yml", "*.yaml", "*.json"]
EXCLUDE_DIRS = ["node_modules", ".git", "venv", "dist", "build", "__pycache__", ".pytest_cache", ".claude", ".playwright-mcp"]

# --- 索引配置 ---
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 128
CHROMA_COLLECTION = "project_knowledge"
