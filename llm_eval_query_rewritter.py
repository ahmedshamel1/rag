"""
multi_llm_query_rewriter_with_judge_and_history.py

Runs multiple LLMs as query rewriters on a set of test questions.
Each LLM keeps its own last 4 messages as history.
A judge LLM (GPT-4.1-mini) evaluates outputs and prints a results table with accuracy and latency.

Author: AI Assistant
"""

import json
import time
from collections import deque
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# -------------------------
# 🔑 API Key (STATIC here, replace with your own)
# -------------------------
OPENROUTER_API_KEY = ""

# -------------------------
# 🤖 Define LLM Builders
# -------------------------
def build_llm(model_name: str, temperature: float = 0.0):
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        model=model_name,
        temperature=temperature,
    )

# ✅ Candidate Models
llms = {
    "llama_8b_3.1": build_llm("meta-llama/llama-3.1-8b-instruct"),
    "llama_3b_3.2": build_llm("meta-llama/llama-3.2-3b-instruct"),
    "llama_1b_3.2": build_llm("meta-llama/llama-3.2-1b-instruct"),
    "mistral_7b": build_llm("mistralai/mistral-7b-instruct"),
    "mistral_3b": build_llm("mistralai/ministral-3b"),
    "flash_1.5_8b": build_llm("google/gemini-flash-1.5-8b"),
    "gemma_4b_3": build_llm("google/gemma-3-4b-it"),
    "gemma_e4b": build_llm("google/gemma-3n-e4b-it"),
}

# history: each model gets its own queue of last 4 messages
histories: Dict[str, deque] = {name: deque(maxlen=4) for name in llms.keys()}

# -------------------------
# 📝 Query Rewriter Prompt (updated with stricter rules to avoid extraneous sections)
# -------------------------
canonical_sections = [
    "ingredients",
    "preparation_time",
    "utensils_needed",
    "preparation_instructions",
    "number_of_servings",
    "nutritional_information",
    "allergen_information",
]

main_prompt_text = (
    "Given the conversation history and a user question, produce EXACTLY one JSON object (and nothing else) "
    "that matches this schema:\n"
    "{{\n"
    "  \"rewritten_query\": string,\n"
    "  \"dish_names\": [string],\n"
    "  \"sections\": [string]\n"
    "}}\n\n"
    "Hard rules (must be followed exactly):\n"
    "1) ALWAYS output exactly one JSON object and nothing else (no explanations, no backticks).\n"
    "2) Use ONLY the following canonical section names when relevant: " + ", ".join(canonical_sections) + ". Do not use any other names.\n"
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

rewriter_prompt = ChatPromptTemplate.from_messages([
    ("system", main_prompt_text),
    ("user", "HISTORY:\n{history}\n\nQUESTION:\n{question}\n\nReturn JSON now."),
])

# -------------------------
# ⚖️ Judge Setup (updated with rules for stricter evaluation, including history)
# -------------------------
judge = build_llm("openai/gpt-4o-mini", temperature=0.0)  # Note: Updated to gpt-4o-mini if available, but was gpt-4.1-mini (assuming typo or variant)

judge_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a strict JSON schema evaluator. "
               "You check if the model output is valid JSON and matches schema "
               "(rewritten_query: string, dish_names: [string], sections: [string]).\n"
               "Canonical sections: " + ", ".join(canonical_sections) + "\n\n"
               "Strict rules for evaluation (must enforce):\n"
               "- dish_names: Must list ONLY dishes explicitly referenced in the question. Use history ONLY if question refers to previous (e.g., 'it', 'last 2'). Do not add extras. Empty if none.\n"
               "- sections: Must be ONLY canonical names and EXACTLY match user request per rules:\n"
               "  - For 'prep' or 'preparation': ONLY ['preparation_time', 'preparation_instructions']\n"
               "  - For 'ingredients': ONLY ['ingredients']\n"
               "  - For full recipe: []\n"
               "  - Map user words to canonical (e.g., 'nutrition facts' -> 'nutritional_information')\n"
               "  - Do NOT allow extra sections; must reflect request precisely.\n"
               "- rewritten_query: Must be concise, non-empty, include relevant dishes/sections from question/history.\n"),
    ("human", "History: {history}\nQuestion: {question}\nOutput: {output}\n\n"
              "Evaluate:\n"
              "- Is it valid JSON with exact schema? (yes/no)\n"
              "- Does dish_names contain exactly the correct dishes based on question and history? (yes/no)\n"
              "- Is rewritten_query reasonable, non-empty, and matching intent based on question/history? (yes/no)\n"
              "- Are sections ONLY canonical and exactly reflect the user request per rules? (yes/no)\n\n"
              "Respond ONLY as JSON with:\n"
              "{{\n"
              "  \"valid_json\": bool,\n"
              "  \"dish_names\": bool,\n"
              "  \"correctness\": bool,\n"
              "  \"valid_sections\": bool\n"
              "}}")
])

# -------------------------
# 🧪 Test Questions
# -------------------------
questions = [
    "What are the prep for refreshing watermelon smoothie?"
]

# -------------------------
# 🏃 Run Experiment
# -------------------------
results: Dict[str, Dict[str, Any]] = {
    name: {
        "valid_json": 0, "dish_names": 0, "correctness": 0, "valid_sections": 0,
        "total": 0, "times": []
    }
    for name in llms.keys()
}

for q in questions:
    print(f"\n🔹 Question: {q}")
    for name, model in llms.items():
        try:
            history_str = "\n".join(histories[name])
            messages = rewriter_prompt.format_messages(history=history_str, question=q)

            # measure response time
            start = time.time()
            output = model.invoke(messages).content.strip()
            end = time.time()
            elapsed = end - start

            print(f"\n{name} Output:\n{output}\n⏱️ Response time: {elapsed:.2f}s")
            
            # Add debugging to see raw output
            print(f"Raw output (repr): {repr(output)}")
            print(f"Output length: {len(output)}")
            print(f"Output starts with: {output[:100] if len(output) > 100 else output}")
            print(f"Output ends with: {output[-100:] if len(output) > 100 else output}")

            histories[name].append(f"User: {q}")
            histories[name].append(f"Model: {output}")

            # judge evaluation
            try:
                judge_messages = judge_prompt.format_messages(history=history_str, question=q, output=output)
                judge_eval = judge.invoke(judge_messages).content
                eval_dict = json.loads(judge_eval)
                print(f"Judge Eval: {eval_dict}")
            except Exception as judge_error:
                print(f"❌ Judge evaluation failed for {name}: {judge_error}")
                print(f"Judge raw output: {repr(judge_eval) if 'judge_eval' in locals() else 'N/A'}")
                # Create a default evaluation dict for failed cases
                eval_dict = {
                    "valid_json": False,
                    "dish_names": False,
                    "correctness": False,
                    "valid_sections": False
                }

            # update stats
            for key in ["valid_json", "dish_names", "correctness", "valid_sections"]:
                if eval_dict.get(key):
                    results[name][key] += 1
            results[name]["total"] += 1
            results[name]["times"].append(elapsed)

        except Exception as e:
            print(f"❌ Error for {name}: {e}")
            # Add better error debugging
            if hasattr(e, '__traceback__'):
                import traceback
                print(f"Full traceback for {name}:")
                traceback.print_exc()
            # Try to show what the model actually returned
            try:
                if 'output' in locals():
                    print(f"Raw output from {name}: {repr(output)}")
            except:
                pass

# -------------------------
# 📊 Results Table
# -------------------------
print("\n===== Evaluation Results =====")
print(f"{'Model':<18} {'ValidJSON%':<10} {'DishNames%':<12} {'Correctness%':<14} {'Sections%':<10} {'AvgTime(s)':<10}")
for name, stats in results.items():
    total = stats["total"] or 1
    avg_time = sum(stats["times"]) / len(stats["times"]) if stats["times"] else 0
    print(f"{name:<18} "
          f"{100*stats['valid_json']/total:<10.1f} "
          f"{100*stats['dish_names']/total:<12.1f} "
          f"{100*stats['correctness']/total:<14.1f} "
          f"{100*stats['valid_sections']/total:<10.1f} "
          f"{avg_time:<10.2f}")