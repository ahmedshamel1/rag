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


from utils.simple_logger import SimpleLogger
from utils.multi_role_document_manager import create_multi_role_document_manager
from utils.query_rewriter import create_query_rewriter
from utils.multi_role_retrievel_manager import _fetch_documents_with_filters

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

doc_manager = create_multi_role_document_manager("multi_role_agent")
from utils.multi_role_document_manager import load_and_index_documents
load_and_index_documents(doc_manager, vector_store)

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
multi_role_logger = SimpleLogger("multi_role", "logs/multi_role_agent_logs.txt")


# Store conversation history manually for query rewriting (since we removed ConversationBufferMemory)
#conversation_history: List[Dict[str, Any]] = []
# Store separate histories for each role
conversation_histories: Dict[str, List[Dict[str, Any]]] = {
    "bakers": [],
    "cofounder": []
}
def _get_role_response(user_input: str, role: str, k_multiplier: int = 1) -> str:
    #global conversation_history
    try:
        
        history = conversation_histories.setdefault(role, [])
        # Rewrite query
        rewrite_result = query_rewriter.rewrite_query(user_input, conversation_history)
        print(f"🔄 Query Rewriter Result: {rewrite_result}")

        if isinstance(rewrite_result, dict):
            rewritten_query = rewrite_result.get("rewritten_query", user_input)
            dish_names = rewrite_result.get("dish_names", []) or []
            sections = rewrite_result.get("sections", []) or []
        else:
            rewritten_query = str(rewrite_result)
            dish_names, sections = [], []

        # Fetch docs with filters based on role
        rag_documents, mechanism_used = _fetch_documents_with_filters(
            rewritten_query, dish_names, sections, role, vector_store
        )
        print(f"📊 Retrieved {len(rag_documents)} documents using {mechanism_used} - Dishes: {dish_names}, Sections: {sections}")

        # Temporary retriever
        original_retriever = llm_chain.retriever
        try:
            if rag_documents:
                temp_retriever = vector_store.as_retriever(
                    search_kwargs={"k": k_multiplier * max(1, len(rag_documents))}
                )
                llm_chain.retriever = temp_retriever
                result = llm_chain.invoke({"query": rewritten_query})
            else:
                result = llm_chain.invoke({"query": rewritten_query})
        finally:
            llm_chain.retriever = original_retriever

        response = result.get("result", str(result))

        # Update history
        #conversation_history.append({"role": "user", "content": user_input})
        #conversation_history.append({"role": "assistant", "content": response})
        #if len(conversation_history) > 6:
        #   conversation_history = conversation_history[-6:]
        # Update history for this role only
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})
        if len(history) > 4:
            conversation_histories[role] = history[-4:]

        # Log interaction
        multi_role_logger.log_interaction(
            user_query=f"Query: {user_input} | Rewritten: {rewritten_query} | Dishes: {dish_names} | Sections: {sections}",
            memory=history,
            rag_data=rag_documents,
            extra_data={
                "original_query": user_input,
                "rewritten_query": rewritten_query,
                "extracted_dish_names": dish_names,
                "extracted_sections": sections,
                "full_recipe_requested": len(sections) == 0,
                "documents_found": len(rag_documents),
                "filter_applied": bool(dish_names or sections),
                "mechanism_used": mechanism_used,
            },
        )

        return response

    except Exception as e:
        print(f"Error in role={role} response: {e}")
        import traceback; traceback.print_exc()
        return "Sorry, I encountered an error while processing your request. Please try again!"


def get_baker_response(user_input: str) -> str:
    """
        Generates a response using direct LLM invocation with a professional baking assistant prompt
        and intelligent query rewriting for follow-up questions.

        Args:
            user_input (str): The user's message or query.

        Returns:
            str: The assistant's response generated based on retrieval and LLM reasoning.
        """
    return _get_role_response(user_input, role="bakers", k_multiplier=7)

def get_cofounder_response(user_input: str) -> str:
    """
    Generates a response using direct LLM invocation with a professional business strategy assistant prompt
    and intelligent query rewriting for follow-up questions. Cofounders can only access cofounder documents.

    Args:
        user_input (str): The user's message or query.

    Returns:
        str: The assistant's response generated based on retrieval and LLM reasoning.
    """
    return _get_role_response(user_input, role="cofounder", k_multiplier=7)