"""
Advanced text splitting utilities using LangChain
"""
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict


class AdvancedTextSplitter:
    """
    LangChain-based text splitter with recursive chunking
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize text splitter
        
        Args:
            chunk_size: Target size of each chunk (in characters, approximating tokens)
            chunk_overlap: Number of overlapping characters
        """
        # Approximate tokens: 1 token ≈ 4 characters
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size * 4,
            chunk_overlap=chunk_overlap * 4,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def split_documents(self, text: str, source: str) -> List[Dict[str, str]]:
        """
        Split text into chunks with metadata
        
        Args:
            text: Input text
            source: Source document name
            
        Returns:
            List of chunk dictionaries
        """
        chunks = self.text_splitter.split_text(text)
        
        return [
            {
                'text': chunk,
                'source': source,
                'chunk_id': f"{source}_chunk_{idx}"
            }
            for idx, chunk in enumerate(chunks)
        ]






