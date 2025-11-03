from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter


def make_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return [chunk for chunk in splitter.split_text(text) if chunk.strip()]
