import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from uuid import uuid4
from chain.core.types import Document
from chain.vectors import FAISSVectorStore
from chain.embeddings import LocalEmbeddings

document_1 = Document(
    page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this morning.",
    metadata={"source": "tweet"},
)

document_2 = Document(
    page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
    metadata={"source": "news"},
)

document_3 = Document(
    page_content="Building an exciting new project with LangChain - come check it out!",
    metadata={"source": "tweet"},
)

document_4 = Document(
    page_content="Robbers broke into the city bank and stole $1 million in cash.",
    metadata={"source": "news"},
)

document_5 = Document(
    page_content="Wow! That was an amazing movie. I can't wait to see it again.",
    metadata={"source": "tweet"},
)

document_6 = Document(
    page_content="Is the new iPhone worth the price? Read this review to find out.",
    metadata={"source": "website"},
)

document_7 = Document(
    page_content="The top 10 soccer players in the world right now.",
    metadata={"source": "website"},
)

document_8 = Document(
    page_content="LangGraph is the best framework for building stateful, agentic applications!",
    metadata={"source": "tweet"},
)

document_9 = Document(
    page_content="The stock market is down 500 points today due to fears of a recession.",
    metadata={"source": "news"},
)

document_10 = Document(
    page_content="I have a bad feeling I am going to get deleted :(",
    metadata={"source": "tweet"},
)

documents = [
    document_1,
    document_2,
    document_3,
    document_4,
    document_5,
    document_6,
    document_7,
    document_8,
    document_9,
    document_10,
]
# Generate UUIDs for your documents
uuids = [str(uuid4()) for _ in range(len(documents))]

# Create vector store and add documents with custom IDs
embeddings = LocalEmbeddings()
vector_store = FAISSVectorStore(embeddings=embeddings)

# Add documents with custom IDs (new feature!)
added_ids = vector_store.add_documents(documents=documents, ids=uuids)
print(f"Added documents with IDs: {added_ids}")

# Search with metadata filtering (new feature!)
results = vector_store.similarity_search(
    "LangChain provides abstractions to make working with LLMs easy",
    k=2,
    filter={"source": "tweet"}
)

for doc, score in results:
    print(f"* {doc.page_content} [{doc.metadata}] (score: {score:.4f})")

# Other new methods you can use:

# # Get documents by IDs
# retrieved_docs = vector_store.get_by_ids([uuids[0]])

# # Get all document IDs
# all_ids = vector_store.get_all_document_ids()
# print(f"All document IDs: {all_ids}")

# # Update a document (metadata/content only, not embeddings)
# vector_store.update_document(uuids[0], Document(
#     page_content="Updated content", 
#     metadata={"source": "documentation", "updated": True}
# ))

# # Delete documents
# vector_store.delete([uuids[0]])

# # Search without scores (just documents)
# docs_only = vector_store.similarity_search_without_scores(
#     "some query", 
#     k=5, 
#     filter={"source": "tweet"}
# )