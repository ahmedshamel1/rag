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


def _find_fuzzy_dish_match(query: str, dish_names: List[str], threshold: int = 80) -> List[str]:
    """Find dish names that fuzzy match the query above the threshold."""
    matched_dishes = []
    query_lower = query.lower()
    
    for dish in dish_names:
        # Check if query contains dish name or vice versa
        if fuzz.partial_ratio(query_lower, dish.lower()) >= threshold:
            matched_dishes.append(dish)
        elif fuzz.token_sort_ratio(query_lower, dish.lower()) >= threshold:
            matched_dishes.append(dish)
    
    return matched_dishes
