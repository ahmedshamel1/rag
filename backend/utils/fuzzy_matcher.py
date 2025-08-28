import os
from typing import List
from fuzzywuzzy import fuzz
import json

def _load_dish_names_from_json() -> List[str]:
    """Load dish names from the JSON file for fuzzy matching."""
    try:
        dish_file = "logs/multi_role_dish_names.json"
        if os.path.exists(dish_file):
            with open(dish_file, 'r') as f:
                return json.load(f)
        return []
    except Exception as e:
        print(f"⚠️ Error loading dish names: {e}")
        return []


def _find_fuzzy_dish_match(dish_names: List[str], available_dishes: List[str], threshold: int = 80) -> List[str]:
    """Find the best match for each dish name from the available dishes. Returns one match per dish name."""
    matched_dishes = []
    
    for dish_name in dish_names:
        dish_lower = dish_name.lower()
        best_match = ""
        best_score = 0
        
        print(f"🔍 DEBUG: Looking for best match for: {dish_name}")
        
        # Find the best matching dish from available dishes
        for available_dish in available_dishes:
            available_lower = available_dish.lower()
            
            # Calculate both similarity scores
            partial_score = fuzz.partial_ratio(dish_lower, available_lower)
            token_score = fuzz.token_sort_ratio(dish_lower, available_lower)
            
            # Use the higher score
            current_score = max(partial_score, token_score)
                        
            # Update best match if this score is higher and meets threshold
            if current_score >= threshold and current_score > best_score:
                best_score = current_score
                best_match = available_dish
        
        # Add the best match for this dish name (or empty string if no match found)
        if best_match:
            print(f"🎯 Final match for '{dish_name}': '{best_match}' (Score: {best_score})")
        else:
            print(f"❌ No match found for '{dish_name}' above threshold {threshold}")
        
        matched_dishes.append(best_match)
    
    return matched_dishes
