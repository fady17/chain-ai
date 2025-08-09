import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from typing import List
from typing_extensions import TypedDict
from langgraph.graph import START, StateGraph

from minichain.core.types import Document
from minichain.rag_runner import create_rag_from_files

# Define state for LangGraph
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def main():
    print("=== Setting up RAG ===")
    # Create RAG runner with your story file
    rag = create_rag_from_files([
        '/Users/fady/Desktop/internship/langchain-clone/chainforge-ai/examples/delete/datasets/story.txt'
    ])
    
    print("\n=== Getting LangGraph Functions ===")
    # Get LangGraph-compatible functions from RAG
    retrieve = rag.get_retrieve_function()
    generate = rag.get_generate_function()
    
    print("\n=== Building LangGraph ===")
    # Build the graph exactly like LangGraph example
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()
    
    print("✅ Graph compiled successfully!")
    
    print("\n=== Visualizing Graph ===")
    # Visualize the graph (saves to file for regular Python)
    try:
        # Try to visualize - will save to file if not in Jupyter
        rag.visualize_graph(save_path="rag_graph.png")
    except Exception as e:
        print(f"Visualization error: {e}")
        print("Graph structure: START → retrieve → generate")
    
    print("\n=== Testing Graph ===")
    # Test the graph with a question
    test_questions = [
        "What is this story about?",
        "Who are the main characters?",
        "What happens in the story?"
    ]
    
    for question in test_questions:
        print(f"\n❓ Question: {question}")
        print("🔄 Processing...")
        
        try:
            # Invoke the graph
            response = graph.invoke({
                "question": question,
                "context": [],
                "answer": ""
            })
            
            print(f"📄 Retrieved {len(response['context'])} documents")
            print(f"💬 Answer: {response['answer']}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n=== Comparison with Direct RAG ===")
    # Compare with direct RAG query
    direct_question = "What is the main theme?"
    
    print(f"\n❓ Question: {direct_question}")
    
    print("\n🔄 Using LangGraph:")
    try:
        langgraph_response = graph.invoke({
            "question": direct_question,
            "context": [],
            "answer": ""
        })
        print(f"Answer: {langgraph_response['answer']}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n🔄 Using Direct RAG:")
    try:
        direct_response = rag.query(direct_question)
        print(f"Answer: {direct_response}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n✅ Example complete!")
    print("📊 Graph visualization saved as 'rag_graph.png' (if supported)")

if __name__ == "__main__":
    main()