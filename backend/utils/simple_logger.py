"""
simple_logger.py

Simple text-based logging system for Multi-Agent RAG Chatbot System.
Logs user queries, conversation memory, and RAG retrieved data to text files.

Author: AI Assistant
Date: August 22, 2025
"""
import os
from datetime import datetime


class SimpleLogger:
    """
    Simple text-based logger for Multi-Agent RAG Chatbot agents.
    
    Logs:
    - User queries
    - Conversation memory
    - RAG retrieved data
    """
    
    def __init__(self, agent_name, log_file_path):
        """
        Initialize the logger for a specific agent.
        
        Args:
            agent_name (str): Name of the agent (e.g., "bakers", "hr", "cofounder")
            log_file_path (str): Path to the text log file
        """
        self.agent_name = agent_name
        self.log_file_path = log_file_path
        self.initialize_log_file()
    
    def initialize_log_file(self):
        """Initialize the text log file with a header."""
        if not os.path.exists(self.log_file_path):
            with open(self.log_file_path, 'w', encoding='utf-8') as f:
                f.write(f"=== {self.agent_name.upper()} AGENT LOGS ===\n")
                f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n\n")
            print(f"Created new log file: {self.log_file_path}")
        else:
            print(f"Log file already exists: {self.log_file_path}")
    
    def log_interaction(self, user_query, memory, rag_data, extra_data=None):
        """
        Log a simple interaction including user query, memory, and RAG data.
        
        Args:
            user_query (str): The user's question
            memory (list): Current conversation memory
            rag_data (list): RAG retrieved documents
            extra_data (dict, optional): Additional structured data to log
        """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                f.write(f"\n--- Interaction at {timestamp} ---\n")
                f.write(f"User Query: {user_query}\n")
                
                # Log memory
                f.write("Conversation Memory:\n")
                if memory:
                    for i, item in enumerate(memory[-3:], 1):  # Last 3 memory items
                        if hasattr(item, 'content'):
                            content = item.content[:100] + "..." if len(item.content) > 100 else item.content
                            f.write(f"  {i}. {content}\n")
                        elif isinstance(item, str):
                            content = item[:100] + "..." if len(item) > 100 else item
                            f.write(f"  {i}. {content}\n")
                        else:
                            f.write(f"  {i}. {str(item)[:100]}...\n")
                else:
                    f.write("  No memory\n")
                
                # Log RAG data
                f.write("RAG Retrieved Data:\n")
                if rag_data:
                    f.write(f"  Total chunks retrieved: {len(rag_data)}\n")
                    for i, doc in enumerate(rag_data, 1):  # Show all retrieved chunks
                        if hasattr(doc, 'page_content'):
                            # Show full content for HR documents, no trimming
                            content = doc.page_content
                            f.write(f"  Chunk {i}: {content}\n")
                            
                            # Only show metadata if it's not empty and contains expected keys
                            if hasattr(doc, 'metadata') and doc.metadata:
                                metadata = doc.metadata
                                f.write(f"    📁 Source: {metadata.get('source_file', 'Unknown')}\n")
                                f.write(f"    🍽️  Dish: {metadata.get('dish_name', 'Unknown')}\n")
                                f.write(f"    📝 Type: {metadata.get('content_type', 'Unknown')}\n")
                                f.write(f"    🔢 Chunk: {metadata.get('chunk_index', '?')}/{metadata.get('total_chunks', '?')}\n")
                                
                                # Add content type emoji for better visualization
                                content_type = metadata.get('content_type', 'Unknown')
                                type_emoji = {
                                    'recipe_name': '🍰',
                                    'ingredients': '🥚',
                                    'preparation_time': '⏰',
                                    'utensils': '🔧',
                                    'preparation_instructions': '📋',
                                    'servings': '👥',
                                    'nutritional_info': '📊',
                                    'allergen_info': '⚠️'
                                }.get(content_type, '📝')
                                f.write(f"    {type_emoji} Section: {type_emoji} {content_type.replace('_', ' ').title()}\n")
                        elif isinstance(doc, str):
                            # Show full string content for HR documents
                            f.write(f"  Chunk {i}: {doc}\n")
                        else:
                            f.write(f"  Chunk {i}: {str(doc)}\n")
                else:
                    f.write("  No chunks retrieved\n")
                
                # Log extra structured data if provided
                if extra_data:
                    f.write("\nQuery Rewriter Analysis:\n")
                    for key, value in extra_data.items():
                        if isinstance(value, list):
                            f.write(f"  {key}: {value}\n")
                        elif isinstance(value, bool):
                            f.write(f"  {key}: {'Yes' if value else 'No'}\n")
                        else:
                            f.write(f"  {key}: {value}\n")
                
                f.write("-" * 50 + "\n")
            
            print(f"Logged interaction for {self.agent_name} agent")
            
        except Exception as e:
            print(f"Error logging interaction: {e}")
    
    def get_log_summary(self):
        """Get a simple summary of the log file."""
        try:
            if not os.path.exists(self.log_file_path):
                return "No log file found"
            
            with open(self.log_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Count interactions (lines starting with "--- Interaction")
            interactions = sum(1 for line in lines if line.startswith("--- Interaction"))
            
            return {
                "total_interactions": interactions
            }
            
        except Exception as e:
            return f"Error getting summary: {e}"

    def hr_log_interaction(self, user_query, memory, rag_data, extra_data=None):
        """
        Log a simple interaction including user query, memory, and RAG data,
        but WITHOUT any document metadata (source file, chunk indices, etc.).
        """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                f.write(f"\n--- Interaction at {timestamp} ---\n")
                f.write(f"User Query: {user_query}\n\n")
                
                # Log memory (last 3 items)
                f.write("Conversation Memory (last 3 items):\n")
                if memory:
                    for i, item in enumerate(memory[-3:], 1):
                        if hasattr(item, 'content'):
                            content = item.content
                            f.write(f"  {i}. {content}\n")
                        elif isinstance(item, str):
                            f.write(f"  {i}. {item}\n")
                        else:
                            f.write(f"  {i}. {str(item)}\n")
                else:
                    f.write("  No memory\n")
                
                f.write("\nRAG Retrieved Data (content only):\n")
                if rag_data:
                    f.write(f"  Total chunks retrieved: {len(rag_data)}\n")
                    for i, doc in enumerate(rag_data, 1):
                        # Prefer page_content when available (LangChain Document)
                        if hasattr(doc, 'page_content'):
                            # Full content for HR documents (no trimming)
                            content = doc.page_content
                            f.write(f"  Chunk {i}:\n{content}\n\n")
                        elif isinstance(doc, str):
                            f.write(f"  Chunk {i}:\n{doc}\n\n")
                        else:
                            f.write(f"  Chunk {i}:\n{str(doc)}\n\n")
                else:
                    f.write("  No chunks retrieved\n\n")
                
                # Log extra structured data if provided (simple key: value)
                if extra_data:
                    f.write("Extra Data:\n")
                    for key, value in (extra_data.items() if isinstance(extra_data, dict) else []):
                        f.write(f"  {key}: {value}\n")
                    f.write("\n")
                
                f.write("-" * 60 + "\n")
            
            print(f"Logged interaction for {getattr(self, 'agent_name', 'agent')} agent (no metadata)")
        
        except Exception as e:
            print(f"Error logging interaction: {e}")
