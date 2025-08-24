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
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
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
hr_docs = load_documents_from_folder("assests/hr_agent")
if hr_docs:
    print(f"📚 Loaded {len(hr_docs)} document chunks for HR Agent")
else:
    print("⚠️ No HR documents found in assests/hr_agent/ folder")

# Creating ChromaDB directory for HR
hr_store = Chroma(persist_directory="database/chroma_hr",
                  embedding_function=embedding_model)  #: Vector store initialized with persistent directory and embedding model

if hr_docs:
    # Clear existing documents and add new ones
    try:
        hr_store.delete_collection()
        hr_store = Chroma(persist_directory="database/chroma_hr",
                         embedding_function=embedding_model)
        hr_store.add_documents(hr_docs)
        print(f"✅ HR documents successfully loaded into vector store")
    except Exception as e:
        print(f"⚠️ Error reloading HR documents: {e}")
        # Try to add documents to existing store
        try:
            hr_store.add_documents(hr_docs)
            print(f"✅ HR documents added to existing vector store")
        except Exception as e2:
            print(f"❌ Failed to add HR documents: {e2}")

# Configure retriever with the same parameters as multi-role agent
hr_retriever = hr_store.as_retriever(
    search_kwargs={
        "k": 7,  # Default number of chunks to retrieve
        "score_threshold": 0.5  # Minimum similarity score threshold
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
                    "k": 7,
                    "score_threshold": 0.5
                }
            )
            hr_chain = ConversationalRetrievalChain.from_llm(llm, hr_retriever, memory=memory_hr,
                                                           output_key="answer")
            
            print(f"✅ Successfully reloaded {len(new_docs)} HR documents")
            return True
        else:
            print("⚠️ No HR documents found to reload")
            return False
            
    except Exception as e:
        print(f"❌ Error reloading HR documents: {e}")
        import traceback
        traceback.print_exc()
        return False

# No SARSA agent - using direct LLM chain invocation

# Initialize simple logger for HR Agent
hr_logger = SimpleLogger("hr", "backend/logs/hr_agent_logs.txt")


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