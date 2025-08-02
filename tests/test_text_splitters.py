# tests/test_text_splitters.py
"""
Test suite for text splitters
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from minichain.text_splitters.implementations import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter
)
from minichain.core.types import Document


def create_sample_text():
    """Create sample text for testing"""
    return """Machine Learning and Artificial Intelligence

Machine learning is a subset of artificial intelligence that focuses on the development of algorithms and statistical models that enable computers to improve their performance on a specific task through experience.

Deep Learning Overview

Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data. It has revolutionized many fields including computer vision, natural language processing, and speech recognition.

Applications in Industry

Machine learning and AI are being applied across various industries:

1. Healthcare: Diagnostic imaging, drug discovery, personalized medicine
2. Finance: Fraud detection, algorithmic trading, risk assessment  
3. Transportation: Autonomous vehicles, route optimization
4. Technology: Recommendation systems, search engines, virtual assistants

The future of AI looks promising with continued advances in computing power, data availability, and algorithmic improvements."""


def test_recursive_character_text_splitter():
    """Test RecursiveCharacterTextSplitter functionality"""
    print("🔄 Testing RecursiveCharacterTextSplitter...")
    
    text = create_sample_text()
    
    # Test 1: Basic splitting
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        length_function=len
    )
    
    chunks = splitter.split_text(text)
    
    print(f"✅ Created {len(chunks)} chunks")
    print(f"✅ First chunk length: {len(chunks[0])}")
    print(f"✅ First chunk preview: {chunks[0][:100]}...")
    
    # Verify chunk sizes
    for i, chunk in enumerate(chunks):
        chunk_length = len(chunk)
        if i < len(chunks) - 1:  # Not last chunk
            assert chunk_length <= splitter.chunk_size + 50, f"Chunk {i} too long: {chunk_length}"
        print(f"   Chunk {i}: {chunk_length} chars")
    
    # Test 2: Different separators
    custom_splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=20,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    custom_chunks = custom_splitter.split_text(text)
    print(f"✅ Custom separators created {len(custom_chunks)} chunks")
    
    # Test 3: Create documents
    documents = splitter.create_documents([text], [{"source": "test_doc", "type": "educational"}])
    
    print(f"✅ Created {len(documents)} Document objects")
    print(f"✅ First document metadata: {documents[0].metadata}")
    
    # Verify metadata
    assert documents[0].metadata["source"] == "test_doc"
    assert documents[0].metadata["chunk_index"] == 0
    assert "total_chunks" in documents[0].metadata
    
    return True


def test_character_text_splitter():
    """Test CharacterTextSplitter functionality"""
    print("\n📄 Testing CharacterTextSplitter...")
    
    text = create_sample_text()
    
    # Test with paragraph separator
    splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=300,
        chunk_overlap=50
    )
    
    chunks = splitter.split_text(text)
    
    print(f"✅ Created {len(chunks)} chunks with paragraph separator")
    for i, chunk in enumerate(chunks):
        print(f"   Chunk {i}: {len(chunk)} chars")
    
    # Test with line separator
    line_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=200,
        chunk_overlap=30
    )
    
    line_chunks = line_splitter.split_text(text)
    print(f"✅ Created {len(line_chunks)} chunks with line separator")
    
    return True


def test_overlap_functionality():
    """Test chunk overlap functionality"""
    print("\n🔗 Testing Overlap Functionality...")
    
    simple_text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence. Sixth sentence."
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=50,
        chunk_overlap=20,
        separators=[". ", " ", ""]
    )
    
    chunks = splitter.split_text(simple_text)
    
    print(f"✅ Created {len(chunks)} chunks from simple text")
    for i, chunk in enumerate(chunks):
        print(f"   Chunk {i}: '{chunk}'")
    
    # Verify overlap exists between consecutive chunks
    if len(chunks) > 1:
        # Check if there's some overlap between chunks
        chunk1_end = chunks[0][-15:]  # Last 15 chars of first chunk
        chunk2_start = chunks[1][:30]  # First 30 chars of second chunk
        
        print(f"✅ Chunk 1 end: '{chunk1_end}'")
        print(f"✅ Chunk 2 start: '{chunk2_start}'")
        
        # There should be some common content
        has_overlap = any(word in chunk2_start for word in chunk1_end.split() if len(word) > 3)
        print(f"✅ Overlap detected: {has_overlap}")
    
    return True


def test_document_splitting():
    """Test splitting existing Document objects"""
    print("\n📑 Testing Document Splitting...")
    
    # Create sample documents
    doc1 = Document(
        page_content=create_sample_text(),
        metadata={"source": "ai_article.txt", "author": "Test Author"}
    )
    
    doc2 = Document(
        page_content="Short document that won't be split.",
        metadata={"source": "short.txt", "type": "note"}
    )
    
    original_docs = [doc1, doc2]
    
    # Split documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    
    split_docs = splitter.split_documents(original_docs)
    
    print(f"✅ Split {len(original_docs)} documents into {len(split_docs)} chunks")
    
    # Verify metadata preservation and enhancement
    for i, doc in enumerate(split_docs[:3]):  # Check first 3
        print(f"   Doc {i} metadata: {doc.metadata}")
        assert "source_document_index" in doc.metadata
        assert "chunk_index" in doc.metadata
        assert "total_chunks" in doc.metadata
    
    # Verify original metadata is preserved
    ai_chunks = [doc for doc in split_docs if doc.metadata.get("source") == "ai_article.txt"]
    assert len(ai_chunks) > 1  # Should be split into multiple chunks
    assert ai_chunks[0].metadata["author"] == "Test Author"
    
    return True


def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n⚠️ Testing Edge Cases...")
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    
    # Test empty text
    empty_chunks = splitter.split_text("")
    print(f"✅ Empty text chunks: {len(empty_chunks)}")
    
    # Test very short text
    short_chunks = splitter.split_text("Short text.")
    print(f"✅ Short text chunks: {len(short_chunks)}")
    assert len(short_chunks) == 1
    
    # Test text exactly at chunk size
    exact_text = "a" * 100
    exact_chunks = splitter.split_text(exact_text)
    print(f"✅ Exact size text chunks: {len(exact_chunks)}")
    
    # Test text much larger than chunk size with no separators
    large_text = "a" * 500
    large_chunks = splitter.split_text(large_text)
    print(f"✅ Large text without separators: {len(large_chunks)} chunks")
    
    return True


def test_integration_with_real_content():
    """Test with more realistic content"""
    print("\n🌍 Testing with Realistic Content...")
    
    # Simulate a research paper or article
    research_content = """
Abstract

This paper presents a comprehensive study of machine learning algorithms and their applications in modern data science. We analyze various supervised and unsupervised learning techniques, comparing their performance across different datasets and use cases.

Introduction

Machine learning has become increasingly important in the digital age. With the exponential growth of data, traditional statistical methods are often insufficient for extracting meaningful insights from large, complex datasets. This research aims to provide a systematic comparison of modern ML algorithms.

Methodology

Our study employed several key approaches:
- Data preprocessing and feature engineering
- Cross-validation techniques for model evaluation  
- Statistical significance testing for performance comparison
- Computational efficiency analysis

We tested algorithms including:
1. Linear Regression and Logistic Regression
2. Decision Trees and Random Forests
3. Support Vector Machines
4. Neural Networks and Deep Learning models
5. Clustering algorithms (K-means, DBSCAN)

Results

The experimental results show significant variations in algorithm performance depending on data characteristics. Linear models performed well on linearly separable data, while ensemble methods showed superior performance on complex, non-linear datasets.

Conclusion

This comprehensive analysis provides valuable insights for practitioners selecting appropriate ML algorithms for their specific use cases. Future work will focus on automated algorithm selection based on dataset characteristics.
    """
    
    # Test with different chunk sizes
    for chunk_size in [200, 500, 1000]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size * 0.1)  # 10% overlap
        )
        
        chunks = splitter.split_text(research_content.strip())
        print(f"✅ Chunk size {chunk_size}: Created {len(chunks)} chunks")
        
        # Verify no chunk is too long
        max_length = max(len(chunk) for chunk in chunks)
        print(f"   Max chunk length: {max_length}")
        assert max_length <= chunk_size + 100  # Allow some flexibility
    
    return True


def run_text_splitter_tests():
    """Run all text splitter tests"""
    print("🧪 Testing Phase 4: Text Splitters")
    print("=" * 60)
    
    tests = [
        test_recursive_character_text_splitter,
        test_character_text_splitter,
        test_overlap_functionality,
        test_document_splitting,
        test_edge_cases,
        test_integration_with_real_content
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed: {e}")
            results.append(False)
    
    if all(results):
        print("\n🎉 Phase 4 Complete - All text splitters working!")
        print("✅ Ready to move to Phase 5: Embeddings")
    else:
        print("\n⚠️ Some text splitter tests failed")
    
    # Show summary
    print(f"\nTest Results: {sum(results)}/{len(results)} passed")
    
    return all(results)


if __name__ == "__main__":
    run_text_splitter_tests()