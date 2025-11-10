"""
PDF Loader utility for extracting text from medical PDFs
"""
import os
from typing import List, Dict
from pathlib import Path
import PyPDF2


class PDFLoader:
    """Handles PDF text extraction and document chunking"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize PDF Loader
        
        Args:
            chunk_size: Maximum number of tokens per chunk
            chunk_overlap: Number of overlapping tokens between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def load_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as string
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            print(f"Error loading PDF {pdf_path}: {e}")
            return ""
    
    def load_txt(self, txt_path: str) -> str:
        """
        Load text from a .txt file
        
        Args:
            txt_path: Path to the text file
            
        Returns:
            File content as string
        """
        try:
            with open(txt_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"Error loading TXT {txt_path}: {e}")
            return ""
    
    def chunk_text(self, text: str, source: str) -> List[Dict[str, str]]:
        """
        Split text into chunks with metadata
        
        Args:
            text: Input text to chunk
            source: Source filename or identifier
            
        Returns:
            List of dictionaries with 'text', 'source', and 'chunk_id'
        """
        # Simple word-based chunking (approximating tokens)
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            
            if chunk_text.strip():  # Only add non-empty chunks
                chunks.append({
                    'text': chunk_text,
                    'source': source,
                    'chunk_id': f"{source}_chunk_{len(chunks)}"
                })
        
        return chunks
    
    def load_and_chunk_document(self, file_path: str) -> List[Dict[str, str]]:
        """
        Load a document and split it into chunks
        
        Args:
            file_path: Path to the document (.pdf or .txt)
            
        Returns:
            List of chunked documents with metadata
        """
        file_ext = Path(file_path).suffix.lower()
        filename = os.path.basename(file_path)
        
        if file_ext == '.pdf':
            text = self.load_pdf(file_path)
        elif file_ext == '.txt':
            text = self.load_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        if not text.strip():
            print(f"Warning: No text extracted from {filename}")
            return []
        
        return self.chunk_text(text, filename)
    
    def load_directory(self, directory_path: str) -> List[Dict[str, str]]:
        """
        Load and chunk all supported documents in a directory
        
        Args:
            directory_path: Path to directory containing documents
            
        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        supported_extensions = ['.pdf', '.txt']
        
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                if Path(file_path).suffix.lower() in supported_extensions:
                    print(f"Processing: {file}")
                    chunks = self.load_and_chunk_document(file_path)
                    all_chunks.extend(chunks)
                    print(f"  - Generated {len(chunks)} chunks")
        
        return all_chunks


def extract_text_from_csv(csv_path: str) -> List[Dict[str, str]]:
    """
    Extract text from CSV file (for QA pairs or structured data)
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        List of text chunks with metadata
    """
    import csv
    
    chunks = []
    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for idx, row in enumerate(reader):
            # Combine all columns into a single text
            text = " | ".join([f"{k}: {v}" for k, v in row.items()])
            chunks.append({
                'text': text,
                'source': os.path.basename(csv_path),
                'chunk_id': f"{os.path.basename(csv_path)}_row_{idx}"
            })
    
    return chunks




