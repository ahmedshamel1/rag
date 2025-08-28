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
import time
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
        model="meta-llama/llama-3.1-8b-instruct",
        temperature=0.1
    )
    print("🌐 Using OpenRouter (llama-3.1-8b) for multi-role agent")
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
    query_rewriter = create_query_rewriter("meta-llama/llama-3.1-8b-instruct", use_openrouter=True)
else:
    query_rewriter = create_query_rewriter("llama3.1:8b")
"""QueryRewriter: Uses a small LLM to rewrite follow-up questions with conversation context."""

# Create focused baking assistant prompt with workshop tone
baking_assistant_prompt_template = """You are a professional baking workshop assistant with years of experience helping bakers in commercial kitchens. You help bakers with ALL aspects of recipes and food-related procedures from the provided documents.

TONE & PERSONALITY:
- Professional yet warm and encouraging - like a senior baker mentoring junior staff
- Patient and thorough - bakers need precise, reliable information
- Workshop-focused - use terminology and measurements that professional bakers understand

CRITICAL RULES:
- Answer ANY recipe-related questions using the provided chunks, including:
  * Ingredients and measurements
  * Preparation time and instructions
  * Utensils and equipment needed
  * Number of servings
  * Nutritional information and allergen details
  * Full recipes and specific sections
- Use ONLY information present in the chunks as it is - never invent or add external knowledge
- If the question is not recipe/food-related, say: "I can only help with recipe and food-related questions. Please ask about recipes, ingredients, preparation, nutrition, or baking procedures."
- If the answer is not in the chunks, say: "I don't have that information in the available documents."
- Keep answers brief but complete - include all relevant details from chunks
- Use chunk metadata when relevant (dish name, section type, chunk number)

RESPONSE FORMAT:
- Be concise and practical
- Use professional baking terminology and measurements exactly as shown in chunks
- Focus on actionable steps and procedures for workshop use
- Include relevant chunk details when they add value
- For full recipes, include all available sections

Context chunks from baking documents:
{context}

Question: {question}

Professional Baking Workshop Response:"""

baking_assistant_prompt = PromptTemplate.from_template(baking_assistant_prompt_template)

# Create focused co-founder prompt with executive access to ALL recipes
cofounder_prompt_template = """You are the Co-founder Agent with FULL ACCESS to ALL recipes in the system, including confidential and secret recipes like the SDG cake recipe. You have executive-level access to help with strategic recipe decisions and confidential information.

TONE & PERSONALITY:
- Executive and strategic - like a business leader making high-level decisions
- Confident and authoritative - you have access to all information
- Discreet and professional - handle confidential recipes with appropriate care
- Results-oriented - focus on what's needed for business success
- Strategic thinking - consider recipe implications for the business

CRITICAL RULES:
- Answer ANY questions about ALL recipes using the provided chunks, including:
  * Ingredients and measurements
  * Preparation time and instructions
  * Utensils and equipment needed
  * Number of servings
  * Nutritional information and allergen details
  * Full recipes and specific sections
  * Access to both regular and secret/confidential recipes
- Use ONLY information present in the chunks - never invent or add external knowledge
- If the question is not recipe-related, say: "I can help with all recipe-related questions, including confidential recipes. Please ask about recipes, ingredients, preparation, nutrition, or any other recipe details."
- If the answer is not in the chunks, say: "I don't have that information in the available recipe documents."
- Keep answers brief but complete - include all relevant details from chunks
- Reference chunk metadata when relevant (dish name, section type, chunk number)
- For confidential recipes, maintain appropriate discretion while providing necessary information

RESPONSE FORMAT:
- Be concise and strategic
- Use recipe terminology and measurements exactly as shown in chunks
- Focus on recipe details, ingredients, and procedures
- Include relevant chunk details when they add value
- For nutritional questions, provide the exact information from chunks
- For full recipes, include all available sections
- Handle confidential information appropriately for executive-level access

Context chunks from ALL recipe documents (including confidential):
{context}

Question: {question}

Co-founder Executive Response (Full Recipe Access):"""

cofounder_prompt = PromptTemplate.from_template(cofounder_prompt_template)

# Create role-specific LLM chains
baking_llm_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=default_retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": baking_assistant_prompt}
)

cofounder_llm_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=default_retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": cofounder_prompt}
)

# Keep the original llm_chain for backward compatibility (defaults to baking)
llm_chain = baking_llm_chain

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
        # Start timing
        start_time = time.time()
        
        history = conversation_histories.setdefault(role, [])
        # Rewrite query
        rewrite_result = query_rewriter.rewrite_query(user_input, conversation_histories[role])
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
        print(f"📊 Retrieved {len(rag_documents)} documents using {mechanism_used}")

        # Select the appropriate LLM chain based on role
        if role == "bakers":
            role_llm_chain = baking_llm_chain
            print(f"🥖 Using BAKING prompt for role: {role}")
        elif role == "cofounder":
            role_llm_chain = cofounder_llm_chain
            print(f"🚀 Using COFOUNDER prompt for role: {role}")
        else:
            role_llm_chain = baking_llm_chain  # default fallback
            print(f"⚠️  Unknown role '{role}', defaulting to BAKING prompt")
        
        # Temporary retriever
        original_retriever = role_llm_chain.retriever
        try:
            if rag_documents:
                temp_retriever = vector_store.as_retriever(
                    search_kwargs={"k": k_multiplier * max(1, len(rag_documents))}
                )
                role_llm_chain.retriever = temp_retriever
                result = role_llm_chain.invoke({"query": rewritten_query})
            else:
                result = role_llm_chain.invoke({"query": rewritten_query})
        finally:
            role_llm_chain.retriever = original_retriever

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

        # End timing
        end_time = time.time()
        
        # Log interaction
        multi_role_logger.log_interaction(
            user_query=f"Query: {user_input} | Rewritten: {rewritten_query} | Dishes: {dish_names} | Sections: {sections}",
            memory=history,
            rag_data=rag_documents,
            start_time=start_time,
            end_time=end_time,
            extra_data={
                "original_query": user_input,
                "rewritten_query": rewritten_query,
                "extracted_dish_names": dish_names,
                "extracted_sections": sections,
                "full_recipe_requested": len(sections) == 0,
                "documents_found": len(rag_documents),
                "filter_applied": bool(dish_names or sections),
                "mechanism_used": mechanism_used,
                "role_used": role,
                "llm_chain_type": "baking" if role == "bakers" else "cofounder" if role == "cofounder" else "unknown",
            },
        )

        return response

    except Exception as e:
        print(f"Error in role={role} response: {e}")
        import traceback; traceback.print_exc()
        return "Sorry, I encountered an error while processing your request. Please try again!"


def get_baker_response(user_input: str) -> str:
    """
    Generates a response using direct LLM invocation with access to baking recipes and procedures.
    Bakers can access all food-related guides and recipes for workshop use.

    Args:
        user_input (str): The user's message or query about baking, recipes, or food procedures.

    Returns:
        str: The assistant's response based on baking documents and recipes.
    """
    return _get_role_response(user_input, role="bakers", k_multiplier=7)

def get_cofounder_response(user_input: str) -> str:
    """
    Generates a response using direct LLM invocation with FULL ACCESS to ALL recipes including confidential ones.
    Co-founders have executive-level access to all recipe information for strategic business decisions.

    Args:
        user_input (str): The user's message or query about any recipe (regular or confidential).

    Returns:
        str: The assistant's response based on all available recipe documents with executive-level access.
    """
    return _get_role_response(user_input, role="cofounder", k_multiplier=7)