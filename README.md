# Mini-Chain

**Mini-Chain** is a micro-framework for building applications with Large Language Models, inspired by LangChain. Its core principle is transparency and modularity, providing a "glass-box" design for engineers who value control and clarity.

## Core Features

- **Modular Components**: Swappable classes for Chat Models, Embeddings, Memory, and more.
- **Local & Cloud Ready**: Supports both local models (via LM Studio) and cloud services (Azure).
- **Modern Tooling**: Built with Pydantic for type-safety and Jinja2 for powerful templating.
- **GPU Acceleration**: Optional `faiss-gpu` support for high-performance indexing.

## Installation

```bash
pip install minichain-ai
#For Local FAISS (CPU) Support:
pip install minichain-ai[local]
#For NVIDIA GPU FAISS Support:
pip install minichain-ai[gpu]
#For Azure Support (Azure AI Search, Azure OpenAI):
pip install minichain-ai[azure]
#To install everything:
pip install minichain-ai[all]
```
Quick Start
Here is the simplest possible RAG pipeline with Mini-Chain:
```bash
pip install minichain-ai[local]

from minichain.rag_runner import create_rag_from_files

# Load knowledge from files
rag = create_rag_from_files(
    file_paths=["path/manual.txt", "README.md"],
    system_prompt="You are a documentation assistant.",
    chunk_size=500,
    retrieval_k=3
)
rag.run_chat()
```
### To Read the full directory
```bash
from minichain.rag_runner import create_rag_from_directory

# Load all Python files from a directory
rag = create_rag_from_directory(
    directory="./src",
    file_extensions=['.py', '.md'],
    system_prompt="You are a code assistant."
)
rag.run_chat()
```

### Custom RAG Configuration
```bash
from minichain.rag_runner import RAGRunner, RAGConfig

config = RAGConfig(
    knowledge_texts=["Your knowledge here..."],
    knowledge_files=["file1.txt", "file2.md"],
    
    # Chunking settings
    chunk_size=1000,
    chunk_overlap=200,
    
    # Retrieval settings
    retrieval_k=4,
    similarity_threshold=0.7,  # Only include high-similarity results
    
    # Chat settings
    system_prompt="Custom system prompt...",
    conversation_keywords=["custom", "keywords", "for", "conversation", "detection"],
    
    # Components (optional - uses defaults if not provided)
    chat_model=None,  # Will use LocalChatModel
    embeddings=None,  # Will use LocalEmbeddings
    text_splitter=None,  # Will use RecursiveCharacterTextSplitter
    vector_store=None,  # Will create FAISSVectorStore
    
    debug=True  # Enable debug output
)

rag = RAGRunner(config).setup()
rag.run_chat()
```
### Using Custom Components
```bash
from minichain.rag_runner import RAGConfig, RAGRunner
from minichain.chat_models import LocalChatModel, LocalChatConfig
from minichain.embeddings import LocalEmbeddings
from minichain.text_splitters import RecursiveCharacterTextSplitter

# Custom components
custom_model = LocalChatModel(LocalChatConfig(temperature=0.7))
custom_embeddings = LocalEmbeddings()
custom_splitter = RecursiveCharacterTextSplitter(chunk_size=800)

config = RAGConfig(
    knowledge_texts=["Your knowledge..."],
    chat_model=custom_model,
    embeddings=custom_embeddings,
    text_splitter=custom_splitter,
)

rag = RAGRunner(config).setup()
rag.run_chat()
```


for azure
pip install minichain-ai[azure]
