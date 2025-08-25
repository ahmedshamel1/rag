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
from backend.utils.simple_logger import SimpleLogger

# Load environment variables from .env file
load_dotenv()

## Document tracking file path
HR_TRACKING_FILE = "backend/logs/hr_agent_document_tracking.json"


def calculate_file_hash(file_path):
    """
    Calculate MD5 hash of a file.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        str: MD5 hash of the file
    """
    try:
        with open(file_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return file_hash
    except Exception as e:
        print(f"Error calculating hash for {file_path}: {e}")
        return None


def load_document_tracking():
    """
    Load the document tracking data from JSON file.
    
    Returns:
        dict: Dictionary mapping filenames to their hashes
    """
    try:
        if os.path.exists(HR_TRACKING_FILE):
            with open(HR_TRACKING_FILE, 'r') as f:
                return json.load(f)
        else:
            return {}
    except Exception as e:
        print(f"Error loading document tracking: {e}")
        return {}


def save_document_tracking(tracking_data):
    """
    Save the document tracking data to JSON file.
    
    Args:
        tracking_data (dict): Dictionary mapping filenames to their hashes
    """
    try:
        os.makedirs(os.path.dirname(HR_TRACKING_FILE), exist_ok=True)
        with open(HR_TRACKING_FILE, 'w') as f:
            json.dump(tracking_data, f, indent=2)
        print(f"✅ Document tracking saved to {HR_TRACKING_FILE}")
    except Exception as e:
        print(f"❌ Error saving document tracking: {e}")


def check_document_changes(folder_path):
    """
    Check for changes in documents by comparing current hashes with stored hashes.
    
    Args:
        folder_path (str): Path to the folder containing documents
        
    Returns:
        tuple: (changed_files, new_files, removed_files)
    """
    current_tracking = load_document_tracking()
    new_tracking = {}
    
    changed_files = []
    new_files = []
    removed_files = []
    
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                current_hash = calculate_file_hash(file_path)
                if current_hash:
                    new_tracking[filename] = current_hash
                    
                    if filename in current_tracking:
                        if current_tracking[filename] != current_hash:
                            changed_files.append(filename)
                            print(f"🔄 Document changed: {filename}")
                    else:
                        new_files.append(filename)
                        print(f"📄 New document detected: {filename}")
    
    # Check for removed files
    for filename in current_tracking:
        if filename not in new_tracking:
            removed_files.append(filename)
            print(f"🗑️ Document removed: {filename}")
    
    # Save new tracking data
    save_document_tracking(new_tracking)
    
    return changed_files, new_files, removed_files


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

def load_documents_from_folder(folder_path):
    """
    Loads all supported documents from the specified folder.
    
    Args:
        folder_path (str): Path to the folder containing documents
        
    Returns:
        list: List of document chunks
    """
    documents = []
    # Use the same chunking parameters as multi-role agent for consistency
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Increased chunk size for better context
        chunk_overlap=50,  # Increased overlap for better continuity
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if filename.lower().endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    # Add role metadata to HR documents
                    for doc in docs:
                        doc.metadata['role'] = 'hr'
                        doc.metadata['source_file'] = filename
                    documents.extend(splitter.split_documents(docs))
                elif filename.lower().endswith('.txt'):
                    loader = TextLoader(file_path, encoding='utf-8')
                    docs = loader.load()
                    # Add role metadata to HR documents
                    for doc in docs:
                        doc.metadata['role'] = 'hr'
                        doc.metadata['source_file'] = filename
                    documents.extend(splitter.split_documents(docs))
                elif filename.lower().endswith('.md'):
                    loader = UnstructuredMarkdownLoader(file_path)
                    docs = loader.load()
                    # Add role metadata to HR documents
                    for doc in docs:
                        doc.metadata['role'] = 'hr'
                        doc.metadata['source_file'] = filename
                    documents.extend(splitter.split_documents(docs))
                elif filename.lower().endswith('.docx'):
                    # Note: Requires python-docx package
                    try:
                        from langchain_community.document_loaders import Docx2txtLoader
                        loader = Docx2txtLoader(file_path)
                        docs = loader.load()
                        # Add role metadata to HR documents
                        for doc in docs:
                            doc.metadata['role'] = 'hr'
                            doc.metadata['source_file'] = filename
                        documents.extend(splitter.split_documents(docs))
                    except ImportError:
                        print(f"python-docx not installed, skipping {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    return documents

# Load documents from HR agent folder
print("🔍 Checking for HR document changes...")
changed_files, new_files, removed_files = check_document_changes("assests/hr_agent")

if changed_files or new_files or removed_files:
    print(f"📊 Document changes detected:")
    if changed_files:
        print(f"  🔄 Changed: {len(changed_files)} files")
    if new_files:
        print(f"  📄 New: {len(new_files)} files")
    if removed_files:
        print(f"  🗑️ Removed: {len(removed_files)} files")
    
    # Load and chunk documents when changes are detected
    hr_docs = load_documents_from_folder("assests/hr_agent")
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


def reload_hr_documents():
    """
    Reloads HR documents from the folder and updates the vector store.
    Useful for updating the knowledge base without restarting the application.
    """
    try:
        print("🔄 Reloading HR documents...")
        global hr_docs, hr_store, hr_retriever, hr_chain
        
        # Check for document changes
        changed_files, new_files, removed_files = check_document_changes("assests/hr_agent")
        
        if changed_files or new_files or removed_files:
            print(f"📊 Document changes detected:")
            if changed_files:
                print(f"  🔄 Changed: {len(changed_files)} files")
            if new_files:
                print(f"  📄 New: {len(new_files)} files")
            if removed_files:
                print(f"  🗑️ Removed: {len(removed_files)} files")
            
            # Load new documents
            new_docs = load_documents_from_folder("assests/hr_agent")
            if new_docs:
                # Clear and recreate the store
                hr_store.delete_collection()
                hr_store = Chroma(persist_directory="database/chroma_hr",
                                 embedding_function=embedding_model)
                hr_store.add_documents(new_docs)
                
                # Update global variables
                hr_docs = new_docs
                hr_retriever = hr_store.as_retriever(
                    search_kwargs={
                        "k": 7
                    }
                )
                hr_chain = ConversationalRetrievalChain.from_llm(llm, hr_retriever, memory=memory_hr,
                                                               output_key="answer")
                
                print(f"✅ Successfully reloaded {len(new_docs)} HR documents")
                return True
            else:
                print("⚠️ No HR documents found to reload")
                return False
        else:
            print("✅ No document changes detected, no reload needed")
            return True
            
    except Exception as e:
        print(f"❌ Error reloading HR documents: {e}")
        import traceback
        traceback.print_exc()
        return False

# No SARSA agent - using direct LLM chain invocation

# Initialize simple logger for HR Agent
hr_logger = SimpleLogger("hr", "backend/logs/hr_agent_logs.txt")
print("📝 HR Agent: Full content logging enabled (no trimming)")


def get_hr_document_stats():
    """
    Get statistics about the HR documents in the vector store.
    
    Returns:
        dict: Dictionary containing document statistics
    """
    try:
        if not hr_docs:
            return {"total_documents": 0, "message": "No HR documents loaded"}
        
        # Count documents by file type
        file_types = {}
        for doc in hr_docs:
            source = doc.metadata.get('source_file', 'Unknown')
            file_ext = source.split('.')[-1].lower() if '.' in source else 'Unknown'
            file_types[file_ext] = file_types.get(file_ext, 0) + 1
        
        # Get total chunks
        total_chunks = len(hr_docs)
        
        # Get vector store info
        collection_count = hr_store._collection.count() if hasattr(hr_store, '_collection') else 0
        
        return {
            "total_documents": total_chunks,
            "vector_store_count": collection_count,
            "file_types": file_types,
            "status": "active"
        }
        
    except Exception as e:
        return {"error": f"Failed to get document stats: {e}"}


def get_document_tracking_stats():
    """
    Get statistics about document tracking and changes.
    
    Returns:
        dict: Dictionary containing document tracking statistics
    """
    try:
        tracking_data = load_document_tracking()
        changed_files, new_files, removed_files = check_document_changes("assests/hr_agent")
        
        return {
            "total_tracked_files": len(tracking_data),
            "changed_files": changed_files,
            "new_files": new_files,
            "removed_files": removed_files,
            "has_changes": bool(changed_files or new_files or removed_files),
            "tracking_file": HR_TRACKING_FILE
        }
        
    except Exception as e:
        return {"error": f"Failed to get tracking stats: {e}"}


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


if __name__ == "__main__":
    """Test the HR agent when run directly."""
    print("🧪 Testing HR Agent with Document Tracking...")
    
    # Print document statistics
    stats = get_hr_document_stats()
    print(f"📊 HR Document Stats: {stats}")
    
    # Print document tracking statistics
    tracking_stats = get_document_tracking_stats()
    print(f"📋 Document Tracking Stats: {tracking_stats}")
    
    # Test a simple query
    test_query = "What are the company policies?"
    print(f"\n🔍 Testing query: '{test_query}'")
    
    try:
        response = hr_qa(test_query)
        print(f"✅ Response: {response[:200]}...")
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    print("\n✅ HR Agent test completed!")