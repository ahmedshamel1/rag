from typing import List, Dict, Any
from utils.fuzzy_matcher import _load_dish_names_from_json, _find_fuzzy_dish_match


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




def _fetch_documents_with_filters(rewritten_query: str, dish_names: List[str], sections: List[str], role: str = "bakers", vector_store=None):
    """Fetch relevant documents from the vector store applying metadata filters.
    
    Implements a three-tier fallback mechanism:
    1. Try with metadata filters + semantic search (with role)
    2. Try fuzzy matching with dish names from JSON + semantic search (with role)
    3. Fall back to semantic search only (with role)

    Returns a tuple of (documents, mechanism_used).
    """
    if vector_store is None:
        print("⚠️ No vector store provided")
        return [], "no vector store"
        
   
    available_dishes = _load_dish_names_from_json()
    # Build a chroma filter with role
    fuzzy_matched_dishes = _find_fuzzy_dish_match(dish_names, available_dishes, threshold=75)

    chroma_filter = _build_chroma_filter(fuzzy_matched_dishes, sections, role)

     # Determine how many chunks to request
    if sections and fuzzy_matched_dishes:
        k = len(sections) * max(1, len(fuzzy_matched_dishes))
    elif dish_names:
        k = 7 * len(fuzzy_matched_dishes)
    else:
        # No sections means full recipe - request more chunks
        k = 7 

    # First mechanism: Try with metadata filters + semantic search (with role)
    if chroma_filter:
        try:
            print(f"🔍 Mechanism 1: Filters + semantic search with role '{role}'")
            documents = vector_store.similarity_search(query=rewritten_query, k=k, filter=chroma_filter)
            print(f"📊 Mechanism 1 returned {len(documents)} documents")
            
            # If we got documents, return them
            if documents:
                return documents, "with content filters"
            else:
                # No documents found with filters, fall back to second mechanism
                print("🔄 Mechanism 1 returned 0 documents, falling back to Mechanism 2: Semantic search only with role...")
                
        except Exception as e:
            print(f"⚠️ Mechanism 1 failed with exception: {e}")
            print("🔄 Falling back to Mechanism 2: Semantic search only with role...")
    
    # # Second mechanism: Try fuzzy matching with dish names from JSON
    # try:
    #     print(f"🔍 Mechanism 2: Fuzzy matching with dish names for role '{role}'")
        
    #     # Load dish names from JSON file
    #     available_dishes = _load_dish_names_from_json()
    #     if available_dishes:
    #         # Try fuzzy matching on the query
    #         fuzzy_matched_dishes = _find_fuzzy_dish_match(dish_names, available_dishes, threshold=75)
            
    #         if fuzzy_matched_dishes:
    #             print(f"🍽️ Fuzzy matched dishes: {fuzzy_matched_dishes}")
    #             # Build filter with fuzzy matched dishes
    #             fuzzy_filter = _build_chroma_filter(fuzzy_matched_dishes, sections, role)
    #             if fuzzy_filter:
    #                 # Use fuzzy matched dishes with semantic search
    #                 k_fuzzy = 7 * len(fuzzy_matched_dishes) if sections else 7
    #                 documents = vector_store.similarity_search(query=rewritten_query, k=k_fuzzy, filter=fuzzy_filter)
    #                 print(f"📊 Mechanism 2 (fuzzy) returned {len(documents)} documents")
    #                 if documents:
    #                     return documents, "fuzzy matching with dish names"
    #             else:
    #                 print("⚠️ No fuzzy filter available")
    #         else:
    #             print("🔍 No fuzzy matches found in query")
    #     else:
    #         print("⚠️ No dish names available for fuzzy matching")
        
    #     print("🔄 Mechanism 2 (fuzzy) returned 0 documents, falling back to Mechanism 3: Semantic search only with role...")
                
    # except Exception as e:
    #     print(f"⚠️ Mechanism 2 (fuzzy) failed with exception: {e}")
    #     print("🔄 Falling back to Mechanism 3: Semantic search only with role...")
    
    # Third mechanism: Fallback to semantic search only (with role)
    try:
        print(f"🔍 Mechanism 3: Semantic search only with role '{role}'")
        # Only apply role-based access control, no content filters
        role_filter = _build_chroma_filter([], [], role)
        
        if role_filter:
            # Use semantic search with role filter only (no dish/section filters)
            # Always use k=7 for fallback mechanism to be more generous
            documents = vector_store.similarity_search(query=rewritten_query, k=7, filter=role_filter)
            print(f"📊 Mechanism 3 (no filters) returned {len(documents)} documents (using k=7)")
            return documents, "semantic search only (role filter only)"
        else:
            # Fallback to semantic search without any filters
            print("⚠️ No role filter available, using semantic search without filters")
            documents = vector_store.similarity_search(query=rewritten_query, k=7)
            print(f"📊 Mechanism 3 (no filters) returned {len(documents)} documents (using k=7)")
            return documents, "semantic search only (no filters)"
            
    except Exception as e:
        print(f"❌ Both mechanisms failed: {e}")
        # Return empty list as last resort
        return [], "no documents found"