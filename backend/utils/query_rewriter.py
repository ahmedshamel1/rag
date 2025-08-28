"""
query_rewriter.py

A dedicated query rewriter that uses a small LLM to reformulate follow-up questions
with conversation context for better retrieval results — returns structured
metadata so the retriever can filter by dish name, sections, and detect full-recipe
requests.

This version uses the LLM alone for extraction (no local heuristics). It will
attempt to parse the LLM's JSON output and will re-prompt once if the first
response is malformed. If the LLM still fails, it returns a safe minimal
fallback dict (empty metadata) so downstream code remains stable.

Author: AI Assistant (updated - LLM-only)
Date: Current
"""
import os
import json
from dotenv import load_dotenv
from typing import List, Dict, Any

from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate

# Load environment variables from .env file
load_dotenv()

# OpenRouter integration
try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class QueryRewriter:
    """
    Query rewriter that relies on an LLM to produce structured JSON output.

    The `rewrite_query` method returns a dict with keys:
      - rewritten_query: str
      - dish_names: List[str]
      - sections: List[str]

    Behaviour:
      1. Call the LLM with a system/user prompt asking for JSON ONLY.
      2. Attempt to parse JSON. If successful, normalize and return.
      3. If parsing fails, re-prompt once with a short strict instruction and the previous output.
      4. If parsing still fails, return a safe minimal fallback dict (no local heuristics).
    """

    CANONICAL_SECTIONS = [
        "ingredients",
        "preparation_time",
        "utensils_needed",
        "preparation_instructions",
        "number_of_servings",
        "nutritional_information",
        "allergen_information"
    ]

    def __init__(self, model_name: str = "llama3.2:1b", use_openrouter: bool = False):
        if use_openrouter and OPENAI_AVAILABLE:
            OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
            if OPENROUTER_API_KEY:
                self.llm = ChatOpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=OPENROUTER_API_KEY,
                    model=model_name,
                    temperature=0.0
                )
            else:
                self.llm = ChatOllama(model=model_name, temperature=0.0)
        else:
            self.llm = ChatOllama(model=model_name, temperature=0.0)

        system_prompt = (
            "You are a strict JSON extractor and query rewriter. "
            "Given the conversation history and a user question, produce EXACTLY one JSON object (and nothing else) "
            "that matches this schema:\n"
            "{{\n"
            "  \"rewritten_query\": string,\n"
            "  \"dish_names\": [string],\n"
            "  \"sections\": [string]\n"
            "}}\n\n"
            "Hard rules (must be followed exactly):\n"
            "1) ALWAYS output exactly one JSON object and nothing else (no explanations, no backticks).\n"
            "2) Use ONLY the following canonical section names when relevant: " + ", ".join(self.CANONICAL_SECTIONS) + ". Do not use any other names.\n"
            "3) If no dish name or section is present or relevant, return an empty list for those fields.\n"
            "4) NEVER include all sections unless explicitly required; only include sections directly matching the user request.\n\n"
            "Rewriting rules (how to build fields):\n"
            "- rewritten_query: a retrieval-ready query string for fetching content. It should include dish names (from history or the question) and, when applicable, section filters (e.g. 'ingredients', 'instructions').\n"
            "- sections: a list of canonical section names that EXACTLY match the user request. Do not add extra sections.\n"
            "- If the user asks for 'prep' or 'preparation', include ONLY 'preparation_time' and 'preparation_instructions'. \n"
            "- If the user asks for 'ingredients', include ONLY 'ingredients'\n"
            "- If the user explicitly asks for a 'full recipe', then and only then return [].\n"
            "- dish_names: list all dishes explicitly referenced in the user question. If the user explicitly requests a single dish, populate dish_names with only that dish (do NOT add other dishes from history as filters).\n\n"
            "Important special-case rules (overrides):\n"
            "- If the user asks for the \"full recipe\" or clearly requests an entire recipe, return an empty sections list [] and make rewritten_query request the full recipe for the dish(es) (e.g. 'full recipe for X').\n"
            "- If the user explicitly names a single dish in their question, do NOT add other dish names from conversation history to dish_names and do NOT add them to rewritten_query. Only include that single dish in dish_names and in the rewritten_query.\n\n"
            "Output requirements:\n"
            "- Rewritten_query should be concise and retrieval-ready.\n"
            "- Do not include any non-canonical section names; map user wording to canonical names where possible.\n"
            "- Output JSON only. No extra text."
        )
        
        # Log the system prompt for debugging


        user_prompt = (
            "CONVERSATION HISTORY: {chat_history}\n"
            "CURRENT QUESTION: {question}\n"
            "Return the JSON object now."
        )

        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", user_prompt)
        ])

    def rewrite_query(self, question: str, chat_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        print(f"🔄 Input Question: '{question}'")
        print(f"🔄 Chat History Length: {len(chat_history)}")
        
        formatted_history = self._format_chat_history(chat_history)

        fallback = {
            "rewritten_query": question,
            "dish_names": [],
            "sections": []
        }

        try:
            # First attempt
            print(f"🔄 Attempting first LLM call...")
            prompt_messages = self.prompt_template.format_messages(
                chat_history=formatted_history,
                question=question
            )
            
            
            response = self.llm.invoke(prompt_messages)
            print(f"🔄 LLM Response received: {type(response)}")
            
            raw = self._extract_raw_content(response)
            parsed = self._try_parse_json(raw)
            if parsed is not None:
                result = self._normalize_parsed(parsed, question)
                print(f"✅ Final Normalized Result: {result}")
                return result

            # Re-prompt once
            print(f"🔄 First attempt failed, trying re-prompt...")
            reprompt_messages = [
                ("system", "Return ONLY a valid JSON object with the schema: {rewritten_query: string, dish_names: [string], sections: [string]}"),
                ("user", f"PREVIOUS OUTPUT: {raw}\nCONVERSATION HISTORY: {formatted_history}\nCURRENT QUESTION: {question}")
            ]
            print(f"🔄 Re-prompt messages: {reprompt_messages}")
            
            reprompt_response = self.llm.invoke(reprompt_messages)
            print(f"🔄 Re-prompt response received: {type(reprompt_response)}")
            
            raw2 = self._extract_raw_content(reprompt_response)
            parsed2 = self._try_parse_json(raw2)
            if parsed2 is not None:
                print(f"✅ Re-prompt succeeded!")
                return self._normalize_parsed(parsed2, question)

            print(f"⚠️ Both attempts failed, using fallback")
            return fallback
        except Exception as e:
            print(f"❌ Exception occurred: {e}")
            print(f"❌ Exception type: {type(e)}")
            import traceback
            print(f"❌ Full traceback:")
            traceback.print_exc()
            print(f"❌ Using fallback due to exception")
            return fallback

    def _format_chat_history(self, chat_history: List[Dict[str, Any]]) -> str:
        if not chat_history:
            return "No previous conversation."
        recent = chat_history[-6:]
        lines = []
        for msg in recent:
            if isinstance(msg, dict):
                role = "Human" if msg.get('role') == 'user' else "Assistant"
                lines.append(f"{role}: {msg.get('content', '')}")
            else:
                role = "Human"
                lines.append(f"{role}: {str(msg)}")
        return "\n".join(lines)

    def _extract_raw_content(self, response: Any) -> str:
        if not response:
            return ""
        
        # Log the raw response structure
        print(f"🔄 Raw LLM Response Object: {type(response)} - {response}")
        
        if isinstance(response, dict):
            raw = response.get('content') or response.get('message') or str(response)
        else:
            raw = getattr(response, 'content', str(response))
        
        print(f"🔄 Extracted Raw Text: '{raw}'")
        return raw

    def _try_parse_json(self, raw_text: str) -> Any:
        if not raw_text:
            return None
        try:
            parsed = json.loads(raw_text)
            print(f"✅ JSON Parsed Successfully: {parsed}")
            return parsed
        except Exception as e:
            print(f"❌ JSON Parse Failed: {e}")
            print(f"   Raw text was: '{raw_text}'")
            return None

    def _normalize_parsed(self, parsed: Any, original_question: str) -> Dict[str, Any]:
        """Normalize parsed JSON to the expected output shape and types.

        Handles a few edge cases the LLM may return:
        - If the LLM returns a JSON array whose first element is an object, use that object.
        - If the parsed value is not a dict (e.g., a string or list without an object), return a safe fallback.
        """
        # If the model returned a list whose first element is an object, use it
        if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
            parsed_obj = parsed[0]
        elif isinstance(parsed, dict):
            parsed_obj = parsed
        else:
            # Not an object we can normalize — return a safe fallback
            return {
                "rewritten_query": original_question,
                "dish_names": [],
                "sections": []
            }

        # Normalize and return
        # Convert dish names to lowercase for case-insensitive filtering
        dish_names = parsed_obj.get("dish_names", []) or []
        dish_names_lower = [d.lower() for d in dish_names if d]
        
        return {
            "rewritten_query": parsed_obj.get("rewritten_query", original_question),
            "dish_names": dish_names_lower,
            "sections": parsed_obj.get("sections", []) or []
        }
def create_query_rewriter(model_name: str = "llama3.2:1b", use_openrouter: bool = False) -> QueryRewriter:
    return QueryRewriter(model_name=model_name, use_openrouter=use_openrouter)
