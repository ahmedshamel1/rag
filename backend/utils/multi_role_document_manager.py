"""
bakers_document_manager.py

Bakers-specific document manager that tracks loaded documents and automatically detects new ones
to prevent re-processing while ensuring all documents are available in ChromaDB.
Specialized for handling baking recipes and related documents.

Author: AI Assistant
Date: August 22, 2025
"""
import os
import json
import hashlib
from typing import List, Dict
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredMarkdownLoader, 
    UnstructuredWordDocumentLoader
)
from langchain.schema import Document


class MultiRoleDocumentManager:
    """
    Manages document loading with tracking to prevent re-processing.
    Handles both bakers and cofounder documents with role-based metadata.
    Loads all documents but only processes new ones.
    """
    
    def __init__(self, tracking_file: str):
        """
        Initialize the document manager.
        
        Args:
            tracking_file (str): Path to JSON file that tracks loaded documents
        """
        self.tracking_file = tracking_file
        self.loaded_documents = self._load_tracking_data()
    
    def _load_tracking_data(self) -> Dict[str, str]:
        """Load the tracking data from JSON file."""
        if os.path.exists(self.tracking_file):
            try:
                with open(self.tracking_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading tracking data: {e}")
                return {}
        return {}
    
    def _save_tracking_data(self):
        """Save the tracking data to JSON file."""
        try:
            os.makedirs(os.path.dirname(self.tracking_file), exist_ok=True)
            with open(self.tracking_file, 'w', encoding='utf-8') as f:
                json.dump(self.loaded_documents, f, indent=2)
        except Exception as e:
            print(f"Error saving tracking data: {e}")
    
    def _get_file_hash(self, file_path: str) -> str:
        """Generate a hash for a file based on its content and modification time."""
        try:
            stat = os.stat(file_path)
            # Combine file size and modification time for hash
            hash_input = f"{stat.st_size}_{stat.st_mtime}"
            return hashlib.md5(hash_input.encode()).hexdigest()
        except Exception as e:
            print(f"Error getting file hash for {file_path}: {e}")
            return ""
    
    def get_new_documents(self, folder_path: str) -> List[Document]:
        """
        Get only new documents that haven't been processed before.
        
        Args:
            folder_path (str): Path to folder containing documents
            
        Returns:
            List[Document]: List of new document chunks
        """
        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            return []
        
        new_documents = []
        new_files = []
        
        print(f"Scanning {folder_path} for new documents...")
        
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                file_hash = self._get_file_hash(file_path)
                
                if file_hash and (filename not in self.loaded_documents or 
                                 self.loaded_documents[filename] != file_hash):
                    # New or modified file
                    try:
                        file_docs = self._load_single_file(file_path, folder_path)
                        if file_docs:
                            new_documents.extend(file_docs)
                            new_files.append(filename)
                            # Update tracking data
                            self.loaded_documents[filename] = file_hash
                            print(f"✅ New/Modified: {filename} ({len(file_docs)} chunks)")
                    except Exception as e:
                        print(f"❌ Error processing {filename}: {e}")
                else:
                    print(f"⏭️  Already processed: {filename}")
        
        if new_files:
            print(f"\n📊 Summary:")
            print(f"  New documents processed: {len(new_files)}")
            print(f"  Total new chunks: {len(new_documents)}")
            print(f"  Total documents tracked: {len(self.loaded_documents)}")
            
            # Save updated tracking data
            self._save_tracking_data()
        else:
            print("✅ No new documents found. All documents are up to date.")
        
        return new_documents
    
    def _load_single_file(self, file_path: str, folder_path: str = "") -> List[Document]:
        """Load a single file based on its extension."""
        filename = os.path.basename(file_path)
        
        try:
            if filename.lower().endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                docs = loader.load()
            elif filename.lower().endswith('.txt'):
                loader = TextLoader(file_path, encoding='utf-8')
                docs = loader.load()
            elif filename.lower().endswith('.md'):
                loader = UnstructuredMarkdownLoader(file_path)
                docs = loader.load()
            elif filename.lower().endswith('.docx'):
                try:
                    loader = UnstructuredWordDocumentLoader(file_path)
                    docs = loader.load()
                except ImportError:
                    print(f"python-docx not installed, skipping {filename}")
                    return []
            else:
                print(f"Unsupported file type: {filename}")
                return []
            
            # Split documents into section-based chunks
            if docs:
                full_content = '\n'.join(d.page_content for d in docs)
                print(f"    📊 Original content length: {len(full_content)} characters")
                full_doc = Document(page_content=full_content)
                chunks = self._split_by_sections(full_doc, filename, folder_path)

                
                # Log chunk details
                for i, chunk in enumerate(chunks):
                    chunk_preview = chunk.page_content[:100] + "..." if len(chunk.page_content) > 100 else chunk.page_content
                    print(f"      Chunk {i+1}: {len(chunk.page_content)} chars - {chunk_preview}")
                
                return chunks
            return []
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            return []
    
    def _split_by_sections(self, doc: Document, filename: str, folder_path: str = "") -> List[Document]:
        """Split the document into chunks based on recipe sections."""
        full_text = doc.page_content
        lines = full_text.splitlines()
        
        if not lines:
            return []
        
        # Auto-detect role based on folder path
        role = 'bakers'  # default
        if 'cofounder' in folder_path.lower():
            role = 'cofounder'
        
        recipe_name = lines[0].strip()
        if not recipe_name:
            # Fallback to filename if first line is empty
            recipe_name = os.path.splitext(filename)[0].replace('_', ' ').replace('-', ' ').title()
        
        section_headers = [
            "Ingredients",
            "Preparation time",
            "Utensils needed",
            "Preparation instructions",
            "Number of servings",
            "Nutritional information (per serving)",
            "Allergen information"
        ]
        
        header_to_type = {
            "Ingredients": "ingredients",
            "Preparation time": "preparation_time",
            "Utensils needed": "utensils_needed",
            "Preparation instructions": "preparation_instructions",
            "Number of servings": "number_of_servings",
            "Nutritional information (per serving)": "nutritional_information",
            "Allergen information": "allergen_information"
        }
        
        section_starts = {}
        for i in range(1, len(lines)):
            stripped = lines[i].strip()
            if not stripped:
                continue
            for header in section_headers:
                if stripped.lower() == header.lower() or stripped.lower().startswith(header.lower() + ':'):
                    if header not in section_starts:
                        section_starts[header] = i
                    break
        
        if len(section_starts) < len(section_headers):
            print(f"    ⚠️  Warning: Only {len(section_starts)}/{len(section_starts)} sections found in {filename}")
        
        sorted_headers = sorted(section_starts, key=section_starts.get)
        
        chunks = []
        for idx, header in enumerate(sorted_headers):
            start = section_starts[header]
            end = section_starts[sorted_headers[idx + 1]] if idx + 1 < len(sorted_headers) else len(lines)
            content = '\n'.join(lines[start:end]).strip()
            
            content_type = header_to_type[header]
            chunk_id = f"{recipe_name.replace(' ', '_').lower()}_{content_type}"
            
            # Convert recipe name to lowercase for case-insensitive filtering
            dish_name_lower = recipe_name.lower()
            
            # Export dish names to JSON file
            self._export_dish_names(dish_name_lower)
            
            metadata = {
                'source_file': filename,
                'dish_name': dish_name_lower,  # Convert to lowercase for case-insensitive filtering
                'content_type': content_type,
                'chunk_id': chunk_id,
                'chunk_index': idx + 1,
                'total_chunks': len(sorted_headers),
                'role': role  # Auto-detected role-based access control
            }
            
            chunk = Document(page_content=content, metadata=metadata)
            chunks.append(chunk)
        
        return chunks
    
    def _export_dish_names(self, dish_name: str):
        """Export dish names to a JSON file for easy access."""
        try:
            dish_file = "logs/multi_role_dish_names.json"
            os.makedirs(os.path.dirname(dish_file), exist_ok=True)
            
            # Load existing dish names or create new list
            existing_dishes = []
            if os.path.exists(dish_file):
                try:
                    with open(dish_file, 'r') as f:
                        existing_dishes = json.load(f)
                except:
                    existing_dishes = []
            
            # Add new dish name if not already present
            if dish_name not in existing_dishes:
                existing_dishes.append(dish_name)
                with open(dish_file, 'w') as f:
                    json.dump(existing_dishes, f, indent=2)
                print(f"    🍽️ Added dish: {dish_name}")
        except Exception as e:
            print(f"    ⚠️ Error exporting dish name: {e}")
    
    def get_all_loaded_files(self) -> List[str]:
        """Get list of all files that have been loaded."""
        return list(self.loaded_documents.keys())
    
    def get_tracking_info(self) -> Dict[str, str]:
        """Get current tracking information."""
        return self.loaded_documents.copy()
    
    def clear_tracking(self):
        """Clear all tracking data (useful for testing)."""
        self.loaded_documents.clear()
        if os.path.exists(self.tracking_file):
            os.remove(self.tracking_file)
        print("Document tracking cleared")



def create_multi_role_document_manager(folder_name: str) -> MultiRoleDocumentManager:
    """
    Create a multi-role document manager for handling documents from multiple folders.
    
    Args:
        folder_name (str): Name of the agent folder (e.g., "bakers_agent")
        
    Returns:
        MultiRoleDocumentManager: Configured multi-role document manager
    """
    tracking_file = f"logs/{folder_name}_document_tracking.json"
    return MultiRoleDocumentManager(tracking_file)

def load_and_index_documents(doc_manager, vector_store) -> int:
    """
    Load new documents for all roles, log detailed chunk information,
    and add them into the vector store.

    Args:
        doc_manager: The MultiRoleDocumentManager instance
        vector_store: The Chroma vector store

    Returns:
        int: Number of new document chunks added
    """
    print("Loading documents for Multi-Role Agent...")
    bakers_docs = doc_manager.get_new_documents("../assests/bakers_agent")
    print("Loading documents for Cofounder Agent...")
    cofounder_docs = doc_manager.get_new_documents("../assests/cofounder_agent")

    new_docs = bakers_docs + cofounder_docs

    if not new_docs:
        print("✅ No new documents to add. All documents are already loaded.")
        return 0

    # Log detailed chunk information
    print(f"\n📋 Detailed Chunk Information:")
    print("=" * 50)
    chunks_by_file = {}
    for doc in new_docs:
        if hasattr(doc, 'metadata') and doc.metadata:
            source_file = doc.metadata.get('source_file', 'Unknown')
            chunks_by_file.setdefault(source_file, []).append(doc)

    for filename, chunks in chunks_by_file.items():
        print(f"\n📁 File: {filename}")
        print(f"  📊 Total chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks, 1):
            metadata = chunk.metadata
            content_type = metadata.get('content_type', 'Unknown')
            dish_name = metadata.get('dish_name', 'Unknown')
            chunk_size = len(chunk.page_content)
            print(f"    Chunk {i}: {content_type} - {dish_name} ({chunk_size} chars)")

    # Add to vector store
    vector_store.add_documents(new_docs)
    print(f"\n✅ Added {len(new_docs)} new document chunks to Multi-Role Agent")

    # Show tracking info
    tracked_files = doc_manager.get_all_loaded_files()
    print(f"📚 Total documents tracked: {len(tracked_files)}")
    if tracked_files:
        print(f"📁 Files: {', '.join(tracked_files[:5])}{'...' if len(tracked_files) > 5 else ''}")

    return len(new_docs)
