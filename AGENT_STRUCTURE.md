# Multi-Agent RAG Chatbot System - New Structure

## 🏗️ **New Agent Architecture**

The system has been restructured to use three specialized agents, each handling multiple document types from their dedicated folders:

### **1. 🥖 Bakers Agent (`bakers_agent.py`)**
- **Purpose**: Specialized in baking-related queries and techniques
- **Document Folder**: `assests/bakers_agent/`
- **Vector Store**: `database/chroma_bakers/`
- **Supported Formats**: PDF, TXT, MD, DOCX
- **Current Documents**: 
  - `ai_book_1.pdf` (moved from original assets)
  - `baking_guide.md` (new sample document)

### **2. 👔 HR Agent (`hr_agent.py`)**
- **Purpose**: Handles HR policies, employee handbook, and workplace procedures
- **Document Folder**: `assests/hr_agent/`
- **Vector Store**: `database/chroma_hr/`
- **Supported Formats**: PDF, TXT, MD, DOCX
- **Current Documents**:
  - `concordia_1.pdf` (moved from original assets)
  - `employee_handbook.txt` (new sample document)

### **3. 🚀 Co-founder Agent (`cofounder_agent.py`)**
- **Purpose**: Specialized in startup strategy, business models, and entrepreneurship
- **Document Folder**: `assests/cofounder_agent/`
- **Vector Store**: `database/chroma_cofounder/`
- **Supported Formats**: PDF, TXT, MD, DOCX
- **Current Documents**:
  - `degree_req.pdf` (moved from original assets)
  - `startup_strategy.md` (new sample document)

## 📁 **Document Management**

### **Adding New Documents**
1. **Place files** in the appropriate agent folder:
   - `assests/bakers_agent/` for baking-related documents
   - `assests/hr_agent/` for HR-related documents
   - `assests/cofounder_agent/` for startup/business documents

2. **Supported file types**:
   - **PDF**: `.pdf` files
   - **Text**: `.txt` files
   - **Markdown**: `.md` files
   - **Word**: `.docx` files (requires `python-docx` package)

3. **Restart the application** after adding new documents

### **Removing Documents**
1. **Delete files** from the agent's folder
2. **Clear vector store** (optional):
   ```python
   # In the agent file, add this function:
   def clear_vector_store():
       vector_store._collection.delete(where={})
   ```
3. **Or delete the entire ChromaDB directory**:
   ```bash
   rm -rf database/chroma_[agent_name]/*
   ```

## 🔄 **Key Changes Made**

### **Removed Features**
- ❌ Wikipedia API integration
- ❌ Web scraping functionality
- ❌ External URL processing

### **Added Features**
- ✅ Multi-document type support (PDF, TXT, MD, DOCX)
- ✅ Dedicated document folders for each agent
- ✅ Automatic document loading from folders
- ✅ Specialized action sets for each agent domain

### **Updated Components**
- **Backend**: `app_server.py` now routes to new agent names
- **Frontend**: `chat_app.py` displays new agent names
- **Database**: New ChromaDB directories for each agent
- **Logging**: New Excel log files for each agent

### **Case-Insensitive Filtering (Latest Update)**
- **Issue Resolved**: ChromaDB filters are case-sensitive, causing filtering failures
- **Solution**: All dish names are automatically converted to lowercase during:
  - Document chunking (in metadata creation)
  - Query filtering (before ChromaDB search)
  - Query rewriting (LLM output normalization)
- **Benefits**: 
  - Consistent filtering regardless of user input case
  - Improved retrieval accuracy
  - Better user experience
- **Note**: Existing documents may need re-processing to ensure lowercase dish names

## 🚀 **Running the System**

### **1. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **2. Start Backend**
```bash
uvicorn app_server:app --reload
```

### **3. Start Frontend**
```bash
streamlit run frontend/chat_app.py
```

### **4. Access Points**
- **Backend API**: `http://127.0.0.1:8000`
- **Frontend UI**: `http://localhost:8501`

## 📊 **Agent Actions**

Each agent now has specialized action sets:

### **Bakers Agent Actions**
- Engage in Baking Conversation
- Retrieve Baking Information
- Provide Baking Tips
- Explain Baking Techniques
- Share Recipe Information
- Offer Baking Advice
- Ask Clarifying Questions
- Show Baking Expertise
- Provide Step-by-Step Instructions
- Share Best Practices
- Explain Ingredients
- Offer Troubleshooting Help
- Share Safety Guidelines
- Provide Equipment Information
- Offer a Summary

### **HR Agent Actions**
- Engage in HR Conversation
- Retrieve HR Information
- Provide Policy Guidance
- Explain Benefits
- Share Company Policies
- Offer HR Advice
- Ask Clarifying Questions
- Show HR Expertise
- Provide Procedure Information
- Share Best Practices
- Explain Employment Terms
- Offer Compliance Help
- Share Safety Guidelines
- Provide Training Information
- Offer a Summary

### **Co-founder Agent Actions**
- Engage in Startup Conversation
- Retrieve Business Information
- Provide Strategic Guidance
- Explain Business Models
- Share Startup Insights
- Offer Business Advice
- Ask Clarifying Questions
- Show Business Expertise
- Provide Strategy Information
- Share Best Practices
- Explain Funding Options
- Offer Growth Advice
- Share Market Insights
- Provide Legal Information
- Offer a Summary

## 🔧 **Technical Details**

### **Document Processing**
- **Chunk Size**: 500 characters
- **Chunk Overlap**: 50 characters
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Store**: ChromaDB with persistent storage

### **Learning System**
- **Algorithm**: SARSA (State-Action-Reward-State-Action)
- **Feedback Integration**: User thumbs up/down ratings
- **Q-Table Storage**: NumPy files for each agent
- **Logging**: Excel files tracking queries, responses, and feedback

### **Memory Management**
- **Conversation Buffer**: Maintains chat history
- **Context Awareness**: Uses previous interactions for better responses
- **Persistent Storage**: ChromaDB maintains document embeddings across sessions

## 📝 **Adding Custom Agents**

To create a new agent:

1. **Create agent file** in `backend/chat_agents/`
2. **Create document folder** in `assests/[agent_name]/`
3. **Create ChromaDB directory** in `database/chroma_[agent_name]/`
4. **Update** `app_server.py` with new routing
5. **Update** `frontend/chat_app.py` with new UI elements
6. **Create log files** in `backend/logs/`

## ⚠️ **Important Notes**

- **Document loading happens at startup** - restart after adding/removing documents
- **ChromaDB is persistent** - changes survive application restarts
- **All agents use the same embedding model** for consistency
- **SARSA learning is agent-specific** - each agent learns independently
- **Memory is conversation-specific** - each agent maintains separate chat history
