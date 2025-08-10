# src/chain/utils/pdf_parser.py
# type: ignore
"""
Lightweight PyMuPDF-based PDF text extraction utility.
Optimized for resumes and short documents.
"""

from pathlib import Path
from typing import Dict, Optional


def extract_pdf_text(pdf_path: Path) -> str:
    """
    Extract text from PDF file using PyMuPDF.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text content as string
        
    Raises:
        ImportError: If PyMuPDF is not installed
        FileNotFoundError: If PDF file doesn't exist
        ValueError: If file is not a valid PDF
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError(
            "PyMuPDF not installed. Install with: pip install pymupdf"
        )
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    try:
        # Open PDF document
        doc = fitz.open(pdf_path)
        
        # Extract text from all pages (resumes are typically 1-2 pages)
        text_parts = []
        for page in doc:
            page_text = page.get_text().strip()
            if page_text:  # Skip empty pages
                text_parts.append(page_text)
        
        # Clean up
        doc.close()
        
        # Join pages with double newlines for clear separation
        return "\n\n".join(text_parts)
        
    except Exception as e:
        raise ValueError(f"Failed to parse PDF {pdf_path}: {str(e)}")


def extract_pdf_with_metadata(pdf_path: Path) -> Dict:
    """
    Extract text and metadata from PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary with 'text' and 'metadata' keys
    """
    try:
        import fitz
    except ImportError:
        raise ImportError(
            "PyMuPDF not installed. Install with: pip install pymupdf"
        )
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    try:
        doc = fitz.open(pdf_path)
        
        # Extract text
        text_parts = []
        for page in doc:
            page_text = page.get_text().strip()
            if page_text:
                text_parts.append(page_text)
        
        # Extract metadata
        metadata = {
            "page_count": doc.page_count,
            "title": doc.metadata.get("title", "").strip(),
            "author": doc.metadata.get("author", "").strip(),
            "creation_date": doc.metadata.get("creationDate", "").strip(),
            "modification_date": doc.metadata.get("modDate", "").strip(),
            "file_size": pdf_path.stat().st_size,
        }
        
        # Clean up empty metadata values
        metadata = {k: v for k, v in metadata.items() if v}
        
        doc.close()
        
        return {
            "text": "\n\n".join(text_parts),
            "metadata": metadata
        }
        
    except Exception as e:
        raise ValueError(f"Failed to parse PDF {pdf_path}: {str(e)}")


def is_pdf_readable(pdf_path: Path) -> bool:
    """
    Quick check if PDF file can be opened and read.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        True if readable, False otherwise
    """
    try:
        import fitz
        doc = fitz.open(pdf_path)
        readable = doc.page_count > 0
        doc.close()
        return readable
    except:
        return False