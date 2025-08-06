"""
Configurable RAG (Retrieval-Augmented Generation) runner for Mini-Chain.

This module provides a high-level interface for creating RAG-enabled chat sessions
with configurable knowledge bases, models, and retrieval strategies.
"""

from typing import List, Dict, Optional, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path

from .core import Document
from .chat_models.base import BaseChatModel
from .chat_models import LocalChatModel, LocalChatConfig
from .embeddings.base import BaseEmbeddings
from .embeddings import LocalEmbeddings
from .vectors.base import BaseVectorStore
from .vectors import FAISSVectorStore
from .text_splitters.base import BaseTextSplitter
from .text_splitters import RecursiveCharacterTextSplitter
from .core.types import SystemMessage, HumanMessage, AIMessage, BaseMessage


@dataclass
class RAGConfig:
    """Configuration class for RAG setup."""
    
    # Knowledge base configuration
    knowledge_texts: List[str] = field(default_factory=list)
    knowledge_files: List[Union[str, Path]] = field(default_factory=list)
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Retrieval configuration
    retrieval_k: int = 3
    similarity_threshold: Optional[float] = None
    
    # Chat configuration
    system_prompt: Optional[str] = None
    conversation_keywords: List[str] = field(default_factory=lambda: [
        "ask", "question", "said", "last", "first", "previous", 
        "conversation", "chat", "tell me about our", "what did"
    ])
    
    # Components (optional - will use defaults if not provided)
    chat_model: Optional[BaseChatModel] = None
    embeddings: Optional[BaseEmbeddings] = None
    text_splitter: Optional[BaseTextSplitter] = None
    vector_store: Optional[BaseVectorStore] = None
    
    # Debug mode
    debug: bool = True


class RAGRunner:
    """
    A configurable RAG runner that handles setup and execution of RAG-enabled chats.
    """
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.vector_store: Optional[BaseVectorStore] = None
        self.chat_model: Optional[BaseChatModel] = None
        
    def setup(self) -> 'RAGRunner':
        """Set up all RAG components based on configuration."""
        if self.config.debug:
            print("--- Setting up RAG Components ---")
        
        # 1. Setup text splitter
        text_splitter = self.config.text_splitter or RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        
        # 2. Setup embeddings
        embeddings = self.config.embeddings or LocalEmbeddings()
        
        # 3. Prepare documents
        documents = self._prepare_documents(text_splitter)
        
        # 4. Setup vector store
        if self.config.vector_store:
            self.vector_store = self.config.vector_store
        else:
            if documents:
                self.vector_store = FAISSVectorStore.from_documents(documents, embeddings)
            else:
                # Create empty vector store
                self.vector_store = FAISSVectorStore(embeddings)
        
        # 5. Setup chat model
        self.chat_model = self.config.chat_model or LocalChatModel(LocalChatConfig())
        
        if self.config.debug:
            print(f"✅ RAG setup complete with {len(documents)} document chunks")
        
        return self
    
    def _prepare_documents(self, text_splitter: BaseTextSplitter) -> List[Document]:
        """Prepare documents from knowledge texts and files."""
        documents = []
        
        # Add knowledge texts
        for text in self.config.knowledge_texts:
            documents.append(Document(page_content=text))
        
        # Add knowledge files
        for file_path in self.config.knowledge_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    documents.append(Document(
                        page_content=content,
                        metadata={"source": str(file_path)}
                    ))
            except Exception as e:
                print(f"Warning: Could not read file {file_path}: {e}")
        
        # Split documents
        if documents:
            split_docs = text_splitter.split_documents(documents)
            return split_docs
        
        return []
    
    def _is_conversation_question(self, user_input: str) -> bool:
        """Determine if the question is about conversation history."""
        return any(keyword in user_input.lower() for keyword in self.config.conversation_keywords)
    
    def _retrieve_context(self, query: str) -> str:
        """Retrieve relevant context for the query."""
        if not self.vector_store:
            return ""
        
        try:
            search_results = self.vector_store.similarity_search(
                query, k=self.config.retrieval_k
            )
            
            context_parts = []
            for result in search_results:
                if isinstance(result, tuple):
                    # Handle (document, score) tuple format
                    doc, score = result
                    if self.config.similarity_threshold is None or score >= self.config.similarity_threshold:
                        context_parts.append(doc.page_content)
                else:
                    # Handle direct document format
                    context_parts.append(result.page_content)
            
            return "\n".join(context_parts)
        except Exception as e:
            if self.config.debug:
                print(f"[DEBUG] Error retrieving context: {e}")
            return ""
    
    def run_chat(self) -> None:
        """Start an interactive RAG-enabled chat session."""
        if not self.chat_model:
            raise RuntimeError("RAG runner not set up. Call setup() first.")
        
        print("\n" + "="*50)
        print(" Mini-Chain RAG Chat ".center(50, " "))
        print("="*50)
        print("Enter your message. Type 'exit' or 'quit' to end the session.")
        
        history: List[Dict[str, str]] = []
        if self.config.system_prompt:
            history.append({"role": "system", "content": self.config.system_prompt})
        
        while True:
            try:
                user_input = input("\n[ You ] -> ")
                if user_input.lower() in ["exit", "quit"]:
                    print("\n🤖 Session ended. Goodbye!")
                    break
                
                # Determine if we need to retrieve context
                if self._is_conversation_question(user_input):
                    if self.config.debug:
                        print("[DEBUG] Conversation question detected - using chat history")
                    enhanced_input = user_input
                else:
                    if self.config.debug:
                        print("[DEBUG] Knowledge question detected - retrieving context")
                    
                    context = self._retrieve_context(user_input)
                    if context:
                        if self.config.debug:
                            print(f"[DEBUG] Retrieved context: {context[:100]}...")
                        enhanced_input = f"Context: {context}\n\nQuestion: {user_input}"
                    else:
                        enhanced_input = user_input
                
                history.append({"role": "user", "content": enhanced_input})
                
                # Convert to message objects
                messages_for_llm: List[BaseMessage] = [
                    SystemMessage(content=msg["content"]) if msg["role"] == "system"
                    else HumanMessage(content=msg["content"]) if msg["role"] == "user"
                    else AIMessage(content=msg["content"])
                    for msg in history
                ]
                
                print("[ AI  ] -> ", end="", flush=True)
                
                # Stream response
                full_response = ""
                for chunk in self.chat_model.stream(messages_for_llm):
                    print(chunk, end="", flush=True)
                    full_response += chunk
                print()  # newline
                
                history.append({"role": "assistant", "content": full_response})
                
                if self.config.debug:
                    print(f"[DEBUG] Conversation has {len(history)} messages")
            
            except KeyboardInterrupt:
                print("\n\n🤖 Session ended. Goodbye!")
                break
            except Exception as e:
                print(f"\nAn error occurred: {e}")
                if self.config.debug:
                    import traceback
                    print(f"[DEBUG] Full traceback: {traceback.format_exc()}")
                break


# Convenience functions for quick setup

def create_rag_from_texts(
    knowledge_texts: List[str],
    system_prompt: Optional[str] = None,
    **kwargs
) -> RAGRunner:
    """Create a RAG runner from a list of knowledge texts."""
    config = RAGConfig(
        knowledge_texts=knowledge_texts,
        system_prompt=system_prompt,
        **kwargs
    )
    return RAGRunner(config).setup()


def create_rag_from_files(
    file_paths: List[Union[str, Path]],
    system_prompt: Optional[str] = None,
    **kwargs
) -> RAGRunner:
    """Create a RAG runner from a list of files."""
    config = RAGConfig(
        knowledge_files=file_paths,
        system_prompt=system_prompt,
        **kwargs
    )
    return RAGRunner(config).setup()


def create_rag_from_directory(
    directory: Union[str, Path],
    file_extensions: List[str] = ['.txt', '.md', '.py'],
    system_prompt: Optional[str] = None,
    **kwargs
) -> RAGRunner:
    """Create a RAG runner from all files in a directory with specified extensions."""
    directory = Path(directory)
    file_paths = []
    
    for ext in file_extensions:
        file_paths.extend(directory.glob(f"**/*{ext}"))
    
    config = RAGConfig(
        knowledge_files=file_paths,
        system_prompt=system_prompt,
        **kwargs
    )
    return RAGRunner(config).setup()
