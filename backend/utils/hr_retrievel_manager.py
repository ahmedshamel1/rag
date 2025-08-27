from langchain_community.vectorstores import Chroma

# Configure retriever
def init_hr_retriever(embedding_model, hr_docs, persist_dir="database/chroma_hr"):
    """
    Initialize the HR vector store and retriever.

    Args:
        embedding_model: The embedding model instance.
        hr_docs (list): List of HR documents to add.
        persist_dir (str): Directory for persisting Chroma DB.

    Returns:
        tuple: (hr_store, hr_retriever)
    """
    hr_store = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_model
    )

    # Add documents if available
    if hr_docs:
        try:
            hr_store.add_documents(hr_docs)
            print(f"✅ HR documents added to vector store")
        except Exception as e:
            print(f"❌ Error adding HR documents: {e}")
    else:
        print("⚠️ No documents to add to vector store")

    hr_retriever = hr_store.as_retriever(
        search_kwargs={"k": 7}  # Default top-k
    )
    return hr_store, hr_retriever


def get_vector_store_status(hr_store, persist_dir="database/chroma_hr"):
    """
    Get the current status of the HR vector store.

    Args:
        hr_store: The Chroma vector store instance.

    Returns:
        dict: Status dictionary
    """
    try:
        has_docs = hr_store._collection.count() > 0 if hasattr(hr_store, '_collection') else False
        if has_docs:
            count = hr_store._collection.count()
            return {
                "status": "active",
                "document_count": count,
                "persist_directory": persist_dir
            }
        else:
            return {
                "status": "empty",
                "document_count": 0,
                "persist_directory": persist_dir
            }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "persist_directory": persist_dir
        }


def fetch_hr_documents(hr_retriever, query: str, k: int = 7):
    """
    Fetch relevant documents from the HR vector store.

    Args:
        hr_retriever: Retriever instance.
        query (str): The user query.
        k (int): Number of documents.

    Returns:
        list: List of retrieved documents.
    """
    try:
        print(f"🔍 HR Retriever: Searching for query: '{query}'")
        documents = hr_retriever.get_relevant_documents(query)
        print(f"📊 Retrieved {len(documents)} documents")

        # Debug output
        for i, doc in enumerate(documents):
            print(f"\n📄 Document {i+1}")
            print(f"Source: {doc.metadata.get('source_file', 'Unknown')}")
            print(f"Content: {doc.page_content[:300]}...")  # preview
            print("-" * 50)

        return documents
    except Exception as e:
        print(f"⚠️ Error retrieving HR documents: {e}")
        return []
