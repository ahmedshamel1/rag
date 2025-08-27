# 🏪 Multi-Agent Bakery Shop RAG System

A sophisticated **multi-agent system** designed for bakery operations, leveraging **Retrieval-Augmented Generation (RAG)** with specialized agents for different business functions. Built using **LangChain**, **OpenRouter** for multi-LLM access, **Ollama** as backup, and **ChromaDB** for intelligent document retrieval and management.

---

## 🎯 System Overview

This intelligent chatbot system serves bakery operations through three specialized agents, each designed to handle domain-specific queries with advanced document retrieval capabilities. The system automatically tracks document changes through hashing mechanisms and employs sophisticated chunking strategies optimized for different content types.

---

## 🤖 Multi-Agent Architecture

### 1. 🧁 **Baker Agent**
Specialized in recipe management, ingredient queries, and baking techniques. Processes structured recipe documents with section-based chunking for optimal retrieval.

### 2. 👔 **Co-Founder Agent**
Handles business strategy, financial planning, and operational decisions. Utilizes the same section-based chunking methodology as the baker agent for consistent document processing.

### 3. 👥 **HR Agent**
Manages employee-related queries, policies, and HR documentation. Employs embedding-based chunking for flexible content retrieval from diverse HR documents.

---

## 🚀 Core Features

- **🔍 Intelligent Document Tracking**: Automatic change detection through document hashing at startup
- **📊 Multi-Tier Retrieval**: Sophisticated fallback mechanisms for optimal chunk selection
- **🎯 Agent-Specific Chunking**: Tailored document processing strategies per agent type
- **🔐 Security Measures**: Built-in prompt templates with security controls
- **🔄 Query Rewriting**: Intelligent query preprocessing for accurate dish name and section identification
- **🐳 Docker Support**: Containerized deployment for easy setup and scaling

---

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **LLM Orchestration** | LangChain | Multi-agent coordination & RAG pipeline |
| **LLM Access** | OpenRouter | Multi-LLM access with single API key |
| **LLM Backup** | Ollama | Local LLM fallback when OpenRouter unavailable |
| **Vector Database** | ChromaDB | Document storage and semantic retrieval |
| **Backend Framework** | FastAPI | High-performance API server |
| **Frontend** | Streamlit | Interactive chat interface |
| **Containerization** | Docker | Consistent deployment environment |

---

## 🧠 Intelligent Document Management

### Document Tracking System
- **Automatic Change Detection**: Documents are hashed at startup to track modifications
- **JSON-Based Logging**: All document changes are logged in structured JSON files
- **Real-Time Updates**: System automatically reloads and reprocesses modified documents

### Agent-Specific Chunking Strategies

#### 🧁 Baker & Co-Founder Agents
- **Section-Based Chunking**: Leverages consistent document structure across recipes
- **Metadata Extraction**: Identifies dish names, sections, and categories
- **Structured Processing**: Optimized for recipe documents with predictable layouts

#### 👥 HR Agent
- **Embedding-Based Chunking**: Uses semantic understanding for flexible content division
- **Top-7 Retrieval**: Returns the 7 most relevant chunks based on semantic similarity
- **Dynamic Content Processing**: Adapts to various HR document formats

---

## 🔍 Advanced Retrieval Mechanisms

### Three-Tier Retrieval System (Baker & Co-Founder)

#### Tier 1: Metadata Filtering
- **Primary Strategy**: Filters by dish names and sections using document metadata
- **High Precision**: Direct matching for exact queries
- **Fast Response**: Minimal computational overhead

#### Tier 2: Fuzzy Matching
- **Fallback Strategy**: Uses Fuzzy Wuzzy algorithm for approximate dish name matching
- **Typo Tolerance**: Handles user input variations and spelling mistakes
- **Confidence Scoring**: Ranks matches by similarity percentage

#### Tier 3: Embedding Retrieval
- **Final Fallback**: Semantic search using vector embeddings
- **Context Understanding**: Captures meaning beyond exact text matches
- **Comprehensive Coverage**: Ensures relevant results even with vague queries

### HR Agent Retrieval
- **Semantic Search**: Leverages embedding models for context-aware retrieval
- **Top-7 Selection**: Returns the most relevant document chunks
- **Dynamic Ranking**: Adapts to query complexity and specificity

---

## 🔧 Query Processing Pipeline

### Query Rewriter
- **Intelligent Preprocessing**: Extracts dish names and sections from user queries
- **Context Enhancement**: Provides structured input to baker and co-founder agents
- **Accuracy Improvement**: Ensures precise metadata filtering

### Security Measures
- **Prompt Templates**: Predefined response patterns for each agent
- **Input Validation**: Sanitizes user queries to prevent injection attacks
- **Access Control**: Agent-specific response limitations and guidelines

---

## 📁 Project Structure

```
rag/
├── 🏪 assests/
│   ├── bakers_agent/          # Recipe and baking documents
│   ├── cofounder_agent/       # Business and strategy documents
│   └── hr_agent/              # HR policies and employee documents
├── 🔧 backend/
│   ├── app_server.py          # FastAPI server entry point
│   ├── chat_agents/           # Agent implementations
│   │   ├── hr_agent.py        # HR agent logic
│   │   └── multi_role_agent.py # Baker & Co-founder agent logic
│   ├── database/              # ChromaDB instances
│   │   ├── chroma_hr/         # HR document embeddings
│   │   └── chroma_multi_role_nomic/ # Recipe embeddings
│   ├── utils/                 # Utility functions
│   │   ├── hr_document_manager.py    # HR document processing
│   │   ├── multi_role_document_manager.py # Recipe processing
│   │   ├── query_rewriter.py  # Query preprocessing
│   │   └── fuzzy_matcher.py   # Fuzzy string matching
│   └── requirements.txt       # Python dependencies
├── 🎨 frontend/
│   ├── chat_app.py            # Streamlit chat interface
│   └── requirements.txt       # Frontend dependencies
├── 🐳 docker-compose.yml      # Multi-container orchestration
├── 📚 docs/                   # API documentation
└── 📝 README.md               # This file
```

---

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.8+ (for local development)
- OpenRouter API key (optional, falls back to Ollama)

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd rag
```

### 2. Environment Setup
```bash
# Copy environment template
cp .env.example .env

# Edit with your configuration
# OPENROUTER_API_KEY=your_key_here
# OLLAMA_BASE_URL=http://localhost:11434
```

### 3. Docker Deployment (Recommended)
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### 4. Local Development
```bash
# Backend setup
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Start backend server
uvicorn app_server:app --reload --host 0.0.0.0 --port 8000

# Frontend setup (new terminal)
cd frontend
pip install -r requirements.txt
streamlit run chat_app.py
```


---

## 📊 API Endpoints

### Chat Endpoints
- `POST /chat/hr` - HR agent queries
- `POST /chat/baker` - Baker agent queries  
- `POST /chat/cofounder` - Co-founder agent queries
- `POST /chat/multi-role` - Multi-role agent queries

### Document Management
- `GET /documents/status` - Document tracking status
- `POST /documents/reload` - Force document reload
- `GET /documents/chunks/{agent}` - View chunked documents

---


## 📈 Performance & Monitoring

### Logging
- **Structured Logging**: JSON-formatted logs for easy parsing
- **Agent-Specific Tracking**: Separate log files for each agent
- **Document Change Monitoring**: Automatic tracking of document modifications

### Metrics
- **Response Time**: Query processing latency tracking
- **Retrieval Accuracy**: Success rates for different retrieval tiers
- **Document Processing**: Chunking and embedding generation statistics

---

## 🔒 Security Features

- **Input Sanitization**: Prevents injection attacks
- **Prompt Templates**: Controlled response generation
- **Access Logging**: Comprehensive audit trails
- **Rate Limiting**: API abuse prevention











