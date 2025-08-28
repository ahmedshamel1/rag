"""
Multi-Agent Virtual Assistant Chat Interface

A Streamlit-based chat interface for interacting with multiple specialized AI agents.
The assistant responses are fetched from a backend API. Styling and layout customizations 
are applied for an improved user experience.
"""

import streamlit as st
import requests
import time
import os

# Configuration
#BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000/process/")
BACKEND_URL = "http://127.0.0.1:8000/process/"  # Replace with your backend API URL


def get_backend_response(prompt: str, mode: str):
    """
    Sends a prompt to the backend API and retrieves the agent's response.

    Args:
        prompt (str): The user's input or query.
        mode (str): The agent type selected (e.g., "general", "ai", "concordia").

    Returns:
        str: The response text from the backend agent, or an error message if the request fails.
    """
    payload = {"user_prompt": prompt, "agent": mode}
    try:
        api_response = requests.post(BACKEND_URL, json=payload)
        api_response.raise_for_status()
        data = api_response.json()
        return data.get("response", "Error: No 'response' key found in backend data.")
    except requests.exceptions.ConnectionError:
        return f"Error: Could not connect to the backend at {BACKEND_URL}. Is it running?"
    except requests.exceptions.RequestException as e:
        return f"Error: Backend request failed: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"


# Streamlit UI

st.set_page_config(page_title="Multi-Chatbot", layout="wide")

# Initialize all session state variables
if "chat_mode_selector" not in st.session_state:
    st.session_state.chat_mode_selector = "Bakers Agent"
if "bakers_chat_history" not in st.session_state:
    st.session_state.bakers_chat_history = [
        {"role": "assistant", "content": "Hi, I'm the Bakers Agent. I can help with all your baking questions!"}]
if "hr_chat_history" not in st.session_state:
    st.session_state.hr_chat_history = [
        {"role": "assistant", "content": "Greetings! I'm the HR Agent. How can I assist with HR matters?"}]
if "cofounder_chat_history" not in st.session_state:
    st.session_state.cofounder_chat_history = [
        {"role": "assistant", "content": "Hello! I'm the Co-founder Agent. Ready to help with startup strategy!"}]
st.markdown("""
    <style>

    /* Style each radio label */
    div[role="radiogroup"] > label {
        border-radius: 8px;
        padding: 0.8em 1.2em;
        margin-bottom: 0.8em;
        cursor: pointer;
        border-left: 4px solid transparent;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        display: flex;
        align-items: center;
        width: 100%;      /* Make all buttons take full width */
        box-sizing: border-box;  /* Include padding in width calculation */
    }

    /* Hover effect */
    div[role="radiogroup"] > label:hover {
        border: 2px solid #ffffff;
    }

    /* Selected item styling */
    div[role="radiogroup"] > label[data-selected="true"] {
        background-color: #d0e4ff !important;
        font-weight: bold;
        border-left: 4px solid #4285f4;
        box-shadow: 0 4px 8px rgba(66,133,244,0.2);
    }

    /* Icon styling */
    div[role="radiogroup"] > label::before {
        margin-right: 10px;
        font-size: 1.2em;
        flex-shrink: 0;  /* Prevent icon from shrinking */
    }
    
    /* Optional: If you want to set a specific fixed width instead of full width */
    /*
    div[role="radiogroup"] {
        display: flex;
        flex-direction: column;
        align-items: center;
    }

    div[role="radiogroup"] > label {
        width: 300px;  /* Set your desired fixed width */
    }
    */
    </style>
""", unsafe_allow_html=True)

st.title("🤖 💬 Multi-Agent Virtual Assistant")

st.sidebar.title("Agent Center")

chat_mode = st.sidebar.radio(
    "",
    ("Bakers Agent", "HR Agent", "Co-founder Agent"),
    index=0,
    key="chat_mode_selector"
)

# Chat history is already initialized in session state above

agent_mode = ""
current_chat_history = ""
# Determine the current chat history and agent mode
if chat_mode == "Bakers Agent":
    current_chat_history = st.session_state.bakers_chat_history
    agent_mode = "bakers"
elif chat_mode == "HR Agent":
    current_chat_history = st.session_state.hr_chat_history
    agent_mode = "hr"
elif chat_mode == "Co-founder Agent":
    current_chat_history = st.session_state.cofounder_chat_history
    agent_mode = "cofounder"

# Display the current chat history
for i, message in enumerate(current_chat_history):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input(f"Ask the '{chat_mode.replace(' Chat', '')}' agent..."):
    current_chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_text = get_backend_response(prompt, agent_mode)
        message_placeholder = st.empty()
        full_response = ""
        for chunk in response_text.split():
            full_response += chunk + " "
            time.sleep(0.08)
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(response_text)
        current_chat_history.append({"role": "assistant", "content": response_text})

    # Update the session state with the new chat history
    if chat_mode == "Bakers Agent":
        st.session_state.bakers_chat_history = current_chat_history
    elif chat_mode == "HR Agent":
        st.session_state.hr_chat_history = current_chat_history
    elif chat_mode == "Co-founder Agent":
        st.session_state.cofounder_chat_history = current_chat_history

   