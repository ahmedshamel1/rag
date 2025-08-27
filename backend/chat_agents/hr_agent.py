"""
HR Agent Module (Document-Based HR Assistant)

This module sets up an HR-focused question-answering system that processes information from
multiple document types stored in the assests/hr_agent/ folder. It integrates a conversational 
retrieval chain using LangChain's LLM, memory, and vector store components.

Key Components:
- Multiple document loaders for PDF, TXT, MD, and DOCX files
- Chroma vector store for semantic search across HR documents
- ConversationalRetrievalChain for context-aware conversations

Used by the HR chatbot in the Streamlit frontend.
Author: Saran Kirthic Sivakumar
Date: April 8, 2025
"""
import os
import hashlib
import json
from dotenv import load_dotenv
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.simple_logger import SimpleLogger
from utils.hr_document_manager import check_document_changes, load_documents_from_folder

# Load environment variables from .env file
load_dotenv()

## Document tracking file path
HR_TRACKING_FILE = "logs/hr_agent_document_tracking.json"


# OpenRouter integration
try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Check for OpenRouter API key, otherwise use Ollama
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if OPENROUTER_API_KEY and OPENAI_AVAILABLE:
    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        model="mistralai/mistral-7b-instruct:free",
        temperature=0.1
    )
    print("🌐 Using OpenRouter (Mistral 7B) for HR agent")
else:
    llm = OllamaLLM(model="llama3.2")
    print("🏠 Using Ollama (llama3.2) for HR agent")

# Use the same embedding model as multi-role agent for consistency
embedding_model = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1.5",
    model_kwargs={"trust_remote_code": True}
    )
print("🔤 Using nomic-embed-text-v1.5 embedding model for HR agent")

memory_hr = ConversationBufferMemory(memory_key="chat_history",
                                    return_messages=True)  #: Memory buffer to enable conversational history in retrieval

# Load documents from HR agent folder
print("🔍 Checking for HR document changes...")
changed_files, new_files, removed_files = check_document_changes("../assests/hr_agent")

if changed_files or new_files or removed_files:
    print(f"📊 Document changes detected:")
    if changed_files:
        print(f"  🔄 Changed: {len(changed_files)} files")
    if new_files:
        print(f"  📄 New: {len(new_files)} files")
    if removed_files:
        print(f"  🗑️ Removed: {len(removed_files)} files")
    
    # Load and chunk documents when changes are detected
    hr_docs = load_documents_from_folder("../assests/hr_agent")
    print(f"📚 Loaded {len(hr_docs)} document chunks for HR Agent")
else:
    print("✅ No document changes detected, skipping document loading")
    hr_docs = []

# Creating ChromaDB directory for HR
hr_store = Chroma(persist_directory="database/chroma_hr",
                  embedding_function=embedding_model)

# Only add documents if we have them
if hr_docs:
    try:
        hr_store.add_documents(hr_docs)
        print(f"✅ HR documents added to vector store")
    except Exception as e:
        print(f"❌ Error adding HR documents: {e}")
else:
    print("⚠️ No documents to add to vector store")

# Configure retriever with the same parameters as multi-role agent
hr_retriever = hr_store.as_retriever(
    search_kwargs={
        "k": 7  # Default number of chunks to retrieve
    }
)
hr_chain = ConversationalRetrievalChain.from_llm(llm, hr_retriever, memory=memory_hr,
                                                 output_key="answer")


# No SARSA agent - using direct LLM chain invocation

# Initialize simple logger for HR Agent
hr_logger = SimpleLogger("hr", "logs/hr_agent_logs.txt")
print("📝 HR Agent: Full content logging enabled (no trimming)")



def get_vector_store_status():
    """
    Get the current status of the HR vector store.
    
    Returns:
        dict: Dictionary containing vector store status
    """
    try:
        has_docs = hr_store._collection.count() > 0 if hasattr(hr_store, '_collection') else False
        if has_docs:
            count = hr_store._collection.count()
            return {
                "status": "active",
                "document_count": count,
                "persist_directory": "database/chroma_hr"
            }
        else:
            return {
                "status": "empty",
                "document_count": 0,
                "persist_directory": "database/chroma_hr"
            }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "persist_directory": "database/chroma_hr"
        }


def _fetch_hr_documents(query: str, k: int = 7):
    """
    Fetch relevant documents from the HR vector store using semantic search.
    
    Args:
        query (str): The user's query
        k (int): Number of documents to retrieve
        
    Returns:
        list: List of relevant documents
    """
    try:
        print(f"🔍 HR Agent: Retrieving documents for query: '{query}'")
        documents = hr_retriever.get_relevant_documents(query)
        print(f"📊 HR Agent retrieved {len(documents)} documents")
        
        # Debug: Show full content of retrieved documents
        for i, doc in enumerate(documents):
            print(f"\n📄 Document {i+1} Content:")
            print(f"Source: {doc.metadata.get('source_file', 'Unknown')}")
            print(f"Content: {doc.page_content}")
            print("-" * 50)
        
        return documents
    except Exception as e:
        print(f"⚠️ Error retrieving HR documents: {e}")
        return []


def hr_qa(prompt):
    """
    Processes user input through the HR agent's conversational pipeline and returns a response.

    Args:
        prompt (str): The user's query or message.

    Returns:
        str: The HR agent's generated response based on retrieval and policy.
    """
    try:
        # Get RAG documents using the dedicated retrieval function
        rag_documents = _fetch_hr_documents(prompt)
        
        # Get current memory
        current_memory = memory_hr.chat_memory.messages if hasattr(memory_hr, 'chat_memory') else []
        
        # Generate response using direct LLM chain invocation
        result = hr_chain.invoke({"question": prompt})
        response = result.get("answer", str(result))
        
        # Log the interaction with detailed information
        hr_logger.log_interaction(
            user_query=prompt,
            memory=current_memory,
            rag_data=rag_documents
        )
        
        print(f"✅ HR Agent: Generated response for query: '{prompt[:50]}...'")
        return response
        
    except Exception as e:
        print(f"❌ Error in hr_qa: {e}")
        import traceback
        traceback.print_exc()
        return "Sorry, I encountered an error while processing your request. Please try again!"

