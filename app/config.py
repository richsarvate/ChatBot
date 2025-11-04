import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_dir: Path
    raw_dir: Path
    processed_dir: Path
    index_dir: Path
    logs_dir: Path
    config_dir: Path
    certs_dir: Path
    regression_cases_path: Path
    openai_api_key: str
    embedding_model: str = "text-embedding-3-large"
    chat_model: str = "gpt-4o"
    chunk_size_tokens: int = 500
    chunk_overlap_tokens: int = 50
    chroma_collection: str = "emails"
    top_k: int = 100  # Retrieve many chunks to ensure keyword matches are included
    top_k_final: int = 10  # Final number after thread deduplication (increased from 6)
    ssl_certfile: Path | None = None
    ssl_keyfile: Path | None = None
    
    # Spam filtering
    spam_penalty: float = 0.3  # Score penalty for low-value automated emails
    spam_subject_patterns: tuple = (
        'order confirmed',
        'order #',
        'automatic reply:',
        'out of office',
        'nightly wrap',
        'your event lineup',
        'top suggestions',
        'recommendations for you',
        'alert:',
        'newsletter',
    )
    spam_sender_patterns: tuple = (
        'no-reply',
        'noreply',
        'donotreply',
        'notifications@',
        'alerts@',
    )


def get_settings() -> Settings:
    load_dotenv()
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data"
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    index_dir = root / "data" / "index" / "chroma"
    logs_dir = root / "logs"
    config_dir = root / "config"
    certs_dir = root / "certs"
    regression_cases_path = root / "tests" / "regression_cases.json"
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env or the environment.")
    
    # SSL configuration
    ssl_certfile = certs_dir / "cert.pem"
    ssl_keyfile = certs_dir / "key.pem"
    
    return Settings(
        project_root=root,
        data_dir=data_dir,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        index_dir=index_dir,
        logs_dir=logs_dir,
        config_dir=config_dir,
        certs_dir=certs_dir,
        regression_cases_path=regression_cases_path,
        openai_api_key=api_key,
        ssl_certfile=ssl_certfile if ssl_certfile.exists() else None,
        ssl_keyfile=ssl_keyfile if ssl_keyfile.exists() else None,
    )
