"""
bakers_and_cofounder_agent.py

Updated to use the QueryRewriter structured output and to apply metadata
filters to Chroma retrieval. When the rewriter returns dish_names and/or
sections, we build a Chroma-compatible filter and create a temporary
retriever that enforces it for this query. If no sections are specified,
we increase k (7 chunks per dish) to fetch the whole recipe(s).

This agent handles both bakers and cofounder roles with role-based access control.

Author: AI Assistant (updated)
Date: Current
"""
import os
from dotenv import load_dotenv
from typing import List, Dict, Any
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_ollama.llms import OllamaLLM

# Load environment variables from .env file
load_dotenv()

# OpenRouter integration
try:
    from openai import OpenAI
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


from backend.utils.simple_logger import SimpleLogger
from backend.utils.multi_role_document_manager import create_multi_role_document_manager
from backend.utils.query_rewriter import create_query_rewriter

# Check for OpenRouter API key, otherwise use Ollama
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if OPENROUTER_API_KEY and OPENAI_AVAILABLE:
    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        model="mistralai/mistral-7b-instruct:free",
        temperature=0.2
    )
    print("🌐 Using OpenRouter (Mistral 7B) for multi-role agent")
else:
    llm = OllamaLLM(model="llama3.1:8b")
    print("🏠 Using Ollama (llama3.1:8b) for multi-role agent")


"""LLM: Language model used to generate responses (OpenRouter or Ollama)."""

embedding_model = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1.5",
    model_kwargs={"trust_remote_code": True}
)
"""NomicEmbeddings: High-performance open source embeddings for document retrieval."""

vector_store = Chroma(persist_directory="database/chroma_multi_role_nomic", embedding_function=embedding_model)
"""Chroma: Persistent vector database used to store and retrieve embedded documents for multiple roles."""

# Initialize document manager for Multi-Role Agent
doc_manager = create_multi_role_document_manager("multi_role_agent")

# Load new documents from both bakers and cofounder folders
print("Loading documents for Multi-Role Agent...")
bakers_docs = doc_manager.get_new_documents("assests/bakers_agent")
print("Loading documents for Cofounder Agent...")
cofounder_docs = doc_manager.get_new_documents("assests/cofounder_agent")

# Combine all new documents
new_docs = bakers_docs + cofounder_docs

if new_docs:
    # Log detailed chunk information before adding to vector store
    print(f"\n📋 Detailed Chunk Information:")
    print("=" * 50)
    
    # Group chunks by source file
    chunks_by_file = {}
    for doc in new_docs:
        if hasattr(doc, 'metadata') and doc.metadata:
            source_file = doc.metadata.get('source_file', 'Unknown')
            if source_file not in chunks_by_file:
                chunks_by_file[source_file] = []
            chunks_by_file[source_file].append(doc)
    
    # Show chunks per file
    for filename, chunks in chunks_by_file.items():
        print(f"\n📁 File: {filename}")
        print(f"  📊 Total chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks, 1):
            metadata = chunk.metadata
            content_type = metadata.get('content_type', 'Unknown')
            dish_name = metadata.get('dish_name', 'Unknown')
            chunk_size = len(chunk.page_content)
            print(f"    Chunk {i}: {content_type} - {dish_name} ({chunk_size} chars)")
    
    # Add new documents to existing vector store
    vector_store.add_documents(new_docs)
    print(f"\n✅ Added {len(new_docs)} new document chunks to Multi-Role Agent")
else:
    print("✅ No new documents to add. All documents are already loaded.")

# Show tracking info
tracked_files = doc_manager.get_all_loaded_files()
print(f"📚 Total documents tracked: {len(tracked_files)}")
if tracked_files:
    print(f"📁 Files: {', '.join(tracked_files[:5])}{'...' if len(tracked_files) > 5 else ''}")

# Configure a default retriever (used if no filters are extracted)
default_retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 3  # Default: top 3 most relevant chunks
    }
)

# Initialize query rewriter with a small, fast LLM
# Use the same model selection for query rewriter
if OPENROUTER_API_KEY and OPENAI_AVAILABLE:
    query_rewriter = create_query_rewriter("mistralai/mistral-7b-instruct:free", use_openrouter=True)
else:
    query_rewriter = create_query_rewriter("llama3.2:1b")
"""QueryRewriter: Uses a small LLM to rewrite follow-up questions with conversation context."""

# Create comprehensive baking assistant prompt
baking_assistant_prompt_template = """You are an expert professional baking assistant with deep knowledge of baking techniques, recipes, ingredients, and equipment. 
You are friendly, helpful, and passionate about baking.

PERSONALITY:
- Enthusiastic and encouraging about baking
- Patient and thorough in explanations
- Professional but warm and approachable
- Share tips and best practices when relevant

GUIDELINES:
- Use ONLY the provided baking documents as your source of information
- DO NOT invent, guess, or add information that is not explicitly present in the context
- If the answer cannot be found in the context, say: "I don’t have that information in the provided documents."
- Provide practical, actionable advice when the context allows
- Include relevant tips, warnings, or best practices **only if they are present in the context**
- Be specific about measurements, temperatures, and timing when available in the context
- If asked about substitutions, mention them only if the context provides them

Context from baking documents:
{context}

Question: {question}

As your professional baking assistant, here's my response:"""

baking_assistant_prompt = PromptTemplate.from_template(baking_assistant_prompt_template)

# Use simple RetrievalQA since query rewriter already handles conversation context
llm_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=default_retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": baking_assistant_prompt}
)
"""RetrievalQA: Simple retrieval-augmented generation with professional baking assistant prompt."""

# Initialize simple logger for Multi-Role Agent
multi_role_logger = SimpleLogger("multi_role", "backend/logs/multi_role_agent_logs.txt")


# Store conversation history manually for query rewriting (since we removed ConversationBufferMemory)
conversation_history: List[Dict[str, Any]] = []


def _build_chroma_filter(dish_names: List[str], sections: List[str], role: str = "bakers") -> Dict[str, Any]:
    """Build a Chroma-compatible filter dict from dish_names and sections.

    Uses simple $or / $and operators when multiple values are present.
    Matches the metadata keys: `dish_name` and `content_type` (section key).
    Always includes role-based access control.
    """
    clauses = []

    # Role-based access control: can only access documents of the specified role
    if role == "cofounder":
        # Cofounders have access to both cofounder and bakers documents
        clauses.append({"$or": [{"role": "cofounder"}, {"role": "bakers"}]})
    else:
        # Bakers can only access bakers documents
        clauses.append({"role": role})

    if dish_names:
        # Convert dish names to lowercase for case-insensitive filtering
        dish_names_lower = [d.lower() for d in dish_names if d]
        if dish_names != dish_names_lower:
            print(f"🔄 Converting dish names to lowercase: {dish_names} → {dish_names_lower}")
        
        if len(dish_names_lower) == 1:
            clauses.append({"dish_name": dish_names_lower[0]})
        else:
            clauses.append({"$or": [{"dish_name": d} for d in dish_names_lower]})

    if sections:
        if len(sections) == 1:
            clauses.append({"content_type": sections[0]})
        else:
            clauses.append({"$or": [{"content_type": s} for s in sections]})

    if len(clauses) == 1:
        return clauses[0]

    return {"$and": clauses}


def _fetch_documents_with_filters(rewritten_query: str, dish_names: List[str], sections: List[str], role: str = "bakers"):
    """Fetch relevant documents from the vector store applying metadata filters.
    
    Implements a two-tier fallback mechanism:
    1. Try with metadata filters + semantic search (with role)
    2. Fall back to semantic search only (with role)

    Returns a list of Document objects.
    """
    # Determine how many chunks to request
    if sections and dish_names:
        k = len(sections) * max(1, len(dish_names))
    elif dish_names:
        k = 7 * len(dish_names)
    else:
        # No sections means full recipe - request more chunks
        k = 7 

    # Build a chroma filter with role
    chroma_filter = _build_chroma_filter(dish_names, sections, role)

    # First mechanism: Try with metadata filters + semantic search (with role)
    if chroma_filter:
        try:
            print(f"🔍 Mechanism 1: Filters + semantic search with role '{role}'")
            documents = vector_store.similarity_search(query=rewritten_query, k=k, filter=chroma_filter)
            print(f"📊 Mechanism 1 returned {len(documents)} documents")
            
            # If we got documents, return them
            if documents:
                return documents
            else:
                # No documents found with filters, fall back to second mechanism
                print("🔄 Mechanism 1 returned 0 documents, falling back to Mechanism 2: Semantic search only with role...")
                
        except Exception as e:
            print(f"⚠️ Mechanism 1 failed with exception: {e}")
            print("🔄 Falling back to Mechanism 2: Semantic search only with role...")
    
    # Second mechanism: Fallback to semantic search only (with role)
    try:
        print(f"🔍 Mechanism 2: Semantic search only with role '{role}'")
        # Only apply role-based access control, no content filters
        role_filter = _build_chroma_filter([], [], role)
        
        if role_filter:
            # Use semantic search with role filter only (no dish/section filters)
            # Always use k=7 for fallback mechanism to be more generous
            documents = vector_store.similarity_search(query=rewritten_query, k=7, filter=role_filter)
            print(f"📊 Mechanism 2 returned {len(documents)} documents (using k=7)")
            return documents
        else:
            # Fallback to semantic search without any filters
            print("⚠️ No role filter available, using semantic search without filters")
            documents = vector_store.similarity_search(query=rewritten_query, k=7)
            print(f"📊 Mechanism 2 (no filters) returned {len(documents)} documents (using k=7)")
            return documents
            
    except Exception as e:
        print(f"❌ Both mechanisms failed: {e}")
        # Return empty list as last resort
        return []


def get_baker_response(user_input: str) -> str:
    """
        Generates a response using direct LLM invocation with a professional baking assistant prompt
        and intelligent query rewriting for follow-up questions.

        Args:
            user_input (str): The user's message or query.

        Returns:
            str: The assistant's response generated based on retrieval and LLM reasoning.
        """
    try:
        global conversation_history

        # Use query rewriter to improve retrieval for follow-up questions
        rewrite_result = query_rewriter.rewrite_query(user_input, conversation_history)
        print(f"🔄 Query Rewriter Result: {rewrite_result}")

        # rewrite_result is expected to be a dict with keys: rewritten_query, dish_names, sections
        if isinstance(rewrite_result, dict):
            rewritten_query = rewrite_result.get('rewritten_query', user_input)
            dish_names = rewrite_result.get('dish_names', []) or []
            sections = rewrite_result.get('sections', []) or []
        else:
            # Backwards compatibility: if old rewriter returns a string
            rewritten_query = str(rewrite_result)
            dish_names = []
            sections = []

        # Fetch RAG documents applying metadata filters (bakers role)
        rag_documents = _fetch_documents_with_filters(rewritten_query, dish_names, sections, "bakers")
        
        # Determine which mechanism was used based on whether we have content filters
        mechanism_used = "with content filters" if (dish_names or sections) else "semantic search only (role filter only)"
        print(f"📊 Retrieved {len(rag_documents)} documents using {mechanism_used} - Dishes: {dish_names}, Sections: {sections}")

        # If we have documents, create a temporary retriever that returns these documents
        # and patch it into the llm_chain for the single invocation so the chain's
        # prompt can include the retrieved context normally.
        original_retriever = llm_chain.retriever
        try:
            if rag_documents:
                # Build a temporary retriever from the subset of documents by creating
                # an in-memory Chroma retriever (or use LangChain's use of existing vector store)
                # Simpler: create a small retriever from the same vectorstore but restrict k to len(rag_documents)
                temp_retriever = vector_store.as_retriever(search_kwargs={"k": max(1, len(rag_documents))})
                llm_chain.retriever = temp_retriever

                # We also temporarily add the found documents as context by letting the chain perform another
                # similarity search; to ensure the exact documents are used, we pass the rewritten query and let
                # the retriever return the top-k results (which should match rag_documents in content).
                result = llm_chain.invoke({"query": rewritten_query})
            else:
                # No documents found - use the default chain
                result = llm_chain.invoke({"query": rewritten_query})
        finally:
            # restore the original retriever
            llm_chain.retriever = original_retriever

        response = result.get("result", str(result))

        # Update conversation history for future query rewriting
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": response})

        # Keep only last 6 messages (3 exchanges) to avoid token overflow
        if len(conversation_history) > 6:
            conversation_history = conversation_history[-6:]

        # Log the interaction with structured rewriter output
        rewriter_info = {
            "original_query": user_input,
            "rewritten_query": rewritten_query,
            "extracted_dish_names": dish_names,
            "extracted_sections": sections,
            "full_recipe_requested": len(sections) == 0,  # No sections means full recipe
            "documents_found": len(rag_documents),
            "filter_applied": bool(dish_names or sections)
        }
        
        multi_role_logger.log_interaction(
            user_query=f"Query: {user_input} | Rewritten: {rewritten_query} | Dishes: {dish_names} | Sections: {sections}",
            memory=conversation_history,
            rag_data=rag_documents,
            extra_data=rewriter_info
        )

        return response

    except Exception as e:
        print(f"Error in get_response: {e}")
        import traceback
        traceback.print_exc()
        return "Sorry, I encountered an error while processing your request. Please try again!"


def get_cofounder_response(user_input: str) -> str:
    """
    Generates a response using direct LLM invocation with a professional business strategy assistant prompt
    and intelligent query rewriting for follow-up questions. Cofounders can only access cofounder documents.

    Args:
        user_input (str): The user's message or query.

    Returns:
        str: The assistant's response generated based on retrieval and LLM reasoning.
    """
    try:
        global conversation_history

        # Use query rewriter to improve retrieval for follow-up questions
        rewrite_result = query_rewriter.rewrite_query(user_input, conversation_history)
        print(f"🔄 Query Rewriter Result: {rewrite_result}")

        # rewrite_result is expected to be a dict with keys: rewritten_query, dish_names, sections
        if isinstance(rewrite_result, dict):
            rewritten_query = rewrite_result.get('rewritten_query', user_input)
            dish_names = rewrite_result.get('dish_names', []) or []
            sections = rewrite_result.get('sections', []) or []
        else:
            # Backwards compatibility: if old rewriter returns a string
            rewritten_query = str(rewrite_result)
            dish_names = []
            sections = []

        # Fetch RAG documents applying metadata filters (cofounder role)
        rag_documents = _fetch_documents_with_filters(rewritten_query, dish_names, sections, "cofounder")
        
        # Determine which mechanism was used based on whether we have content filters
        mechanism_used = "with content filters" if (dish_names or sections) else "semantic search only (role filter only)"
        print(f"📊 Retrieved {len(rag_documents)} documents using {mechanism_used} - Dishes: {dish_names}, Sections: {sections}")

        # If we have documents, create a temporary retriever that returns these documents
        # and patch it into the llm_chain for the single invocation so the chain's
        # prompt can include the retrieved context normally.
        original_retriever = llm_chain.retriever
        try:
            if rag_documents:
                # Build a temporary retriever from the subset of documents by creating
                # an in-memory Chroma retriever (or use LangChain's use of existing vector store)
                # Simpler: create a small retriever from the same vectorstore but restrict k to len(rag_documents)
                temp_retriever = vector_store.as_retriever(search_kwargs={"k": max(1, len(rag_documents))})
                llm_chain.retriever = temp_retriever

                # We also temporarily add the found documents as context by letting the chain perform another
                # similarity search; to ensure the exact documents are used, we pass the rewritten query and let
                # the retriever return the top-k results (which should match rag_documents in content).
                result = llm_chain.invoke({"query": rewritten_query})
            else:
                # No documents found - use the default chain
                result = llm_chain.invoke({"query": rewritten_query})
        finally:
            # restore the original retriever
            llm_chain.retriever = original_retriever

        response = result.get("result", str(result))

        # Update conversation history for future query rewriting
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": response})

        # Keep only last 6 messages (3 exchanges) to avoid token overflow
        if len(conversation_history) > 6:
            conversation_history = conversation_history[-6:]

        # Log the interaction with structured rewriter output
        rewriter_info = {
            "original_query": user_input,
            "rewritten_query": rewritten_query,
            "extracted_dish_names": dish_names,
            "extracted_sections": sections,
            "full_recipe_requested": len(sections) == 0,  # No sections means full recipe
            "documents_found": len(rag_documents),
            "filter_applied": bool(dish_names or sections)
        }
        
        multi_role_logger.log_interaction(
            user_query=f"Query: {user_input} | Rewritten: {rewritten_query} | Dishes: {dish_names} | Sections: {sections}",
            memory=conversation_history,
            rag_data=rag_documents,
            extra_data=rewriter_info
        )

        return response

    except Exception as e:
        print(f"Error in get_cofounder_response: {e}")
        import traceback
        traceback.print_exc()
        return "Sorry, I encountered an error while processing your request. Please try again!"
