import os
import json
import hashlib
from typing import List
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredMarkdownLoader, 
    UnstructuredWordDocumentLoader
)
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

## Document tracking file path
HR_TRACKING_FILE = "logs/hr_agent_document_tracking.json"


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
                else:
                    print(f"⚠️ Failed to calculate hash for: {filename}")
    else:
        print(f"❌ Folder does not exist: {folder_path}")
    
    # Only check for removed files if we had previous tracking data
    # If current_tracking is empty, all files are new, not removed
    if current_tracking:  # Only check for removed files if we had previous data
        for filename in current_tracking:
            if filename not in new_tracking:
                removed_files.append(filename)
                print(f"🗑️ Document removed: {filename}")
    
    # Save new tracking data
    save_document_tracking(new_tracking)
    
    return changed_files, new_files, removed_files


def reload_hr_documents():
    """
    Reloads HR documents from the folder and updates the vector store.
    Useful for updating the knowledge base without restarting the application.
    """
    try:
        print("🔄 Reloading HR documents...")
        global hr_docs, hr_store, hr_retriever, hr_chain
        
        # Check for document changes
        changed_files, new_files, removed_files = check_document_changes("../assests/hr_agent")
        
        if changed_files or new_files or removed_files:
            print(f"📊 Document changes detected:")
            if changed_files:
                print(f"  🔄 Changed: {len(changed_files)} files")
            if new_files:
                print(f"  📄 New: {len(new_files)} files")
            if removed_files:
                print(f"  🗑️ Removed: {len(removed_files)} files")
            
            # Load new documents
            new_docs = load_documents_from_folder("../assests/hr_agent")
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
        changed_files, new_files, removed_files = check_document_changes("../assests/hr_agent")
        
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
