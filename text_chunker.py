from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter

class TextChunker:
    def __init__(self, chunk_size=10000, chunk_overlap=1000):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def chunk_text(self, text: str) -> List[str]:
        return self.splitter.split_text(text)

