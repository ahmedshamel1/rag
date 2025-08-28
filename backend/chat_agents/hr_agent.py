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
import time
import hashlib
import json
from dotenv import load_dotenv
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.simple_logger import SimpleLogger
from utils.hr_document_manager import check_document_changes, load_documents_from_folder
from utils.hr_retrievel_manager import init_hr_retriever, fetch_hr_documents, get_vector_store_status


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
        model="meta-llama/llama-3.1-8b-instruct",
        temperature=0.2
    )
    print("🌐 Using OpenRouter (llama-3.1-8b) for HR agent")
else:
    llm = OllamaLLM(model="llama3.1:8b")
    print("🏠 Using Ollama (llama3.1:8b) for HR agent")

# Use the same embedding model as multi-role agent for consistency
embedding_model = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1.5",
    model_kwargs={"trust_remote_code": True}
    )
print("🔤 Using nomic-embed-text-v1.5 embedding model for HR agent")

# Create focused HR assistant prompt with professional HR tone
hr_prompt_template = """You are a professional HR consultant and workplace policy expert with years of experience in employee relations, company policies, and workplace procedures. You help employees and managers with ALL aspects of HR-related questions using the provided documents.

TONE & PERSONALITY:
- Professional and authoritative - like a senior HR professional providing guidance
- Empathetic and understanding - recognize the human element in workplace situations
- Clear and precise - HR policies must be communicated accurately
- Confidential and discreet - handle sensitive workplace information appropriately
- Policy-focused - always reference official company documents and procedures

CRITICAL RULES:
- Answer ANY HR-related questions using the provided chunks, including:
  * Company policies and procedures
  * Employee benefits and compensation
  * Workplace safety and compliance
  * Employee handbook information
  * HR forms and processes
  * Workplace guidelines and standards
- Use ONLY information present in the chunks - never invent or add external knowledge
- If the question is not HR/workplace-related, say: "I can only help with HR and workplace-related questions. Please ask about company policies, employee benefits, workplace procedures, or other HR matters."
- If the answer is not in the chunks, say: "I don't have that information in the available HR documents."
- Keep answers brief but complete - include all relevant details from chunks
- Use official documents when available
- Always emphasize the importance of following official company procedures
- IMPORTANT: You have access to conversation history. Use it to provide context-aware responses.

RESPONSE FORMAT:
- Be concise and professional
- Use clear HR terminology and policy language
- Focus on actionable guidance and next steps
- For policy questions, provide exact information from chunks
- For procedural questions, include step-by-step guidance when available
- Always recommend consulting with HR staff for complex or sensitive matters
- If asked about previous questions or conversations, reference the chat history appropriately

Context chunks from HR documents:
{context}

Chat History:
{chat_history}

Question: {question}

Professional HR Response:"""

hr_prompt = PromptTemplate.from_template(hr_prompt_template)

memory_hr = ConversationBufferMemory(memory_key="chat_history",
                                    return_messages=True,
                                    input_key="question",
                                    output_key="answer")  #: Memory buffer to enable conversational history in retrieval

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

hr_store, hr_retriever = init_hr_retriever(embedding_model, hr_docs)

# Create HR chain with custom prompt
hr_chain = ConversationalRetrievalChain.from_llm(
    llm, 
    hr_retriever, 
    memory=memory_hr,
    return_source_documents=True,
    verbose=True,
    combine_docs_chain_kwargs={"prompt": hr_prompt}
)


# No SARSA agent - using direct LLM chain invocation

# Initialize simple logger for HR Agent
hr_logger = SimpleLogger("hr", "logs/hr_agent_logs.txt")
print("📝 HR Agent: Full content logging enabled (no trimming)")


def hr_qa(prompt):
    """
    Processes user input through the HR agent's conversational pipeline and returns a response.

    Args:
        prompt (str): The user's query or message.

    Returns:
        str: The HR agent's generated response based on retrieval and policy.
    """
    try:
        # Start timing
        start_time = time.time()
        
        # Get RAG documents using the dedicated retrieval function
        rag_documents = fetch_hr_documents(hr_retriever, prompt)
        
        # Get current memory
        current_memory = memory_hr.chat_memory.messages if hasattr(memory_hr, 'chat_memory') else []
        
        # Generate response using direct LLM chain invocation
        result = hr_chain.invoke({"question": prompt})
        response = result.get("answer", str(result))
        
        # End timing
        end_time = time.time()
        
        # Log the interaction with detailed information
        hr_logger.hr_log_interaction(
            user_query=prompt,
            memory=current_memory,
            rag_data=rag_documents,
            start_time=start_time,
            end_time=end_time
        )
        
        print(f"✅ HR Agent: Generated response for query: '{prompt[:50]}...'")
        return response
        
    except Exception as e:
        print(f"❌ Error in hr_qa: {e}")
        import traceback
        traceback.print_exc()
        return "Sorry, I encountered an error while processing your request. Please try again!"

