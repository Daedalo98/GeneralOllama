import streamlit as st
import functions
import requests
import os
from datetime import datetime
import json
from pathlib import Path



# !! need to distribute these logs !!
'''
functions.add_log("---log App initialization started ---") 
functions.add_log("---log New chat button clicked ---") 
functions.add_log("---log Chat session started ---") 
functions.add_log("---log Chat model selected: GPT-3.5 ---") 
functions.add_log("---log Saved chat file loaded ---") 
functions.add_log("---log Ollama connection established (URL: http://example.com) ---") 
functions.add_log("---log Generation parameters set: temperature=0.7, max_tokens=512, top_p=0.9, top_k=15 ---") 
functions.add_log("---log Streaming mode enabled: True ---") 
functions.add_log("---log Chat messages displayed ---") 
functions.add_log("---log Chat saved successfully ---") 
functions.add_log("---log UI elements rendered ---")
'''
# !!


def chat_json_path(chat_name):
    """Return a safe JSON filename for a chat."""
    safe_name = Path(chat_name).name  # prevent names containing directory paths
    return os.path.join(CHAT_FOLDER, f"{safe_name}.json")

prompt_names = list(st.session_state.prompts_dict.keys())
ollama_models = st.session_state.ollama_models
OLLAMA_HOST = st.session_state.ollama_host
CHAT_FOLDER = st.session_state.folder_paths["CHAT_FOLDER"]

with st.sidebar:
    # ---------------------------------------------------------
    # Chat settings
    # ---------------------------------------------------------
    with st.container(border=False):             
        with st.expander("🧠 CHAT model settings", expanded=True):
            # model selection
            text_idx = ollama_models.index("granite4.2:3b") if "granite4.2:3b" in ollama_models else 0
            MODEL = st.selectbox("Main Model", ollama_models, index=text_idx, key="Ollama chat model")

            st.subheader("Load a saved chat")
            # Only offer chats that have a matching JSON file.
            saved_chat_files = sorted(Path(CHAT_FOLDER).glob("*.json"))

            if saved_chat_files:
                selected_chat = st.selectbox(
                    "Saved chats",
                    options=saved_chat_files,
                    format_func=lambda path: path.stem,
                    key="saved_chat_selector",
                )

                if st.button("Load Chat", key="load_chat_button"):
                    try:
                        with open(selected_chat, "r", encoding="utf-8") as f:
                            chat_data = json.load(f)

                        # Basic validation prevents malformed files from breaking the app.
                        messages = chat_data.get("messages", [])
                        if not isinstance(messages, list):
                            raise ValueError("The chat file does not contain a message list.")

                        st.session_state.chat_messages = messages
                        st.session_state.chat_system_prompt = chat_data.get("system_prompt", "")
                        st.session_state.chat_model = chat_data.get("model", MODEL)

                        started_at = chat_data.get("started_at")
                        st.session_state.chat_started_at = (
                            datetime.fromisoformat(started_at) if started_at else datetime.now()
                        )

                        # Mark the chat as active if your app uses this flag.
                        st.session_state.chat_active = True
                        st.rerun()

                    except (OSError, json.JSONDecodeError, ValueError) as e:
                        st.error(f"Could not load that chat: {e}")
            else:
                st.caption("No reloadable chats found.")

            with st.expander("Chat advanced settings", expanded=False, icon="⚙️"):
                
                # Ollama's local HTTP API.
                OLLAMA_URL = st.text_input("CHAT Ollama URL", value=f"{OLLAMA_HOST}/api/chat")

                # Generation temperature.
                # CHAT should generally be deterministic, so 0 is appropriate.
                TEMPERATURE = st.number_input("CHAT Temperature", 0.0, 3.0, 0.6, 0.1, help="Higher values make output more random. For CHAT, 0 is recommended for deterministic output.")

                # Maximum number of generated tokens.
                MAX_TOKENS = st.number_input("CHAT Max Tokens", -1, 100000, 10000, 1, help="Maximum number of tokens to generate. -1 = no limit.")

                # HTTP timeout.
                REQUEST_TIMEOUT = st.number_input("CHAT Request Timeout", 1, 1000, 300, 1, help="Timeout for CHAT requests in seconds.")

                STREAMING = st.toggle("Streaming Generation", value=True, key="chat_streaming", help="Whether to stream the model's output as it generates. If disabled, the model will return the full response at once.")

                TOP_P = st.number_input("Top-p", 0.0, 1.0, 0.9, 0.01, help="Top-p sampling for CHAT generation. Lower values = more focused output.")
                TOP_K = st.number_input("Top-k", 1, 1000, 50, 1, help="Top-k sampling for CHAT generation. Lower values = more focused output.")

            # path to save the chat
            CHAT_NAME = st.text_input(
                "Name the chat (without extension) before saving it!", 
                value=f"{st.session_state.chat_started_at}_with_{MODEL}", 
                help=f"by default, chats will be stored as .md files in {CHAT_FOLDER}")
            
            st.subheader("System Prompt")
            default_index = (
                prompt_names.index("CHAT_PROMPT")
                if "CHAT_PROMPT" in prompt_names else 0)

            prompt_name = st.selectbox(
                "Active Prompt",
                options=prompt_names,
                index=default_index,
                key="chat_prompt_selector"
            )

            chat_prompt_text = st.session_state.prompts_dict.get(prompt_name,"")
            # show the prompt
            st.text(chat_prompt_text)

# =========================================================
# CHAT SESSION RENDERING    
# =========================================================

if st.session_state.chat_start:
    # don't re-enter
    st.session_state.chat_start = False

    # Delete any previous conversation from memory.
    st.session_state.chat_messages = []

    # Record when this conversation started.
    st.session_state.chat_started_at = datetime.now()

    # IMPORTANT:
    # Snapshot the prompt selected at the moment the chat is started.
    st.session_state.chat_system_prompt = chat_prompt_text

    # Also remember the selected model.
    st.session_state.chat_model = MODEL

    functions.add_log(f"chat started with model: {MODEL}")
    st.rerun()

# =========================================================
# MAIN CHAT
# =========================================================
st.title("💬 Chat with your Ollama LLM")

# Display previous messages, including both user and assistant messages
for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            thinking = message.get("thinking", "")
            if thinking:
                with st.expander("🧠 Reasoning", expanded=False):
                    st.markdown(thinking)

        st.markdown(message.get("content", ""))


# Chat input
user_prompt = st.chat_input(
    f"Message {MODEL}...", 
    submit_mode="stop", 
    disabled=not st.session_state.chat_active,
    #accept_file="multiple",
    #file_type=[".md", ".txt"],
    )
if user_prompt:
    # Store the user's message
    user_message = {
        "role": "user",
        "content": user_prompt,
    }

    st.session_state.chat_messages.append(user_message)

    # Immediately show the user's message.
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Build the messages sent to Ollama
    ollama_messages = [
        {
            "role": "system",
            "content": st.session_state.chat_system_prompt
        }    
    ]

    # # Add the actual conversation
    for message in st.session_state.chat_messages:
        ollama_messages.append(
            {
                "role": message["role"],
                "content": message["content"],
            }
        )

    # Prepare assistant UI
    with st.chat_message("assistant"):
        # Reasoning display
        thinking_container = st.expander("🧠 Reasoning", expanded=False)
        with thinking_container:
            thinking_placeholder = st.empty()

        # Answer display
        answer_placeholder = st.empty()

        # Accumulators
        thinking_text = ""
        answer_text = ""

        try: # Stream Ollama
            for event in functions.stream_ollama_chat(
                messages=ollama_messages,
                ollama_url=OLLAMA_URL,
                model=MODEL,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                timeout=REQUEST_TIMEOUT,
                top_p=TOP_P,
                top_k=TOP_K,
                thinking=True, # Enable Ollama thinking.
                ):

                event_type = event["type"]
                chunk = event.get("content", "")

                # THINKING
                if event_type == "thinking":
                    thinking_text += chunk

                    # Update the reasoning area.
                    thinking_placeholder.markdown(thinking_text)

                # FINAL ANSWER
                elif event_type == "content":
                    answer_text += chunk

                    if STREAMING:
                        answer_placeholder.markdown(answer_text+ "▌")

                elif event_type == "done":
                    break

            # Render final answer
            answer_placeholder.markdown(answer_text)

            # Store complete assistant message
            assistant_message = {
                "role": "assistant",
                "content": answer_text,
                "thinking": thinking_text,
            }

            st.session_state.chat_messages.append(assistant_message)

        except requests.exceptions.Timeout:
            error_message = ("⚠️ The request to Ollama timed out.")
            answer_placeholder.error(error_message)

        except requests.exceptions.RequestException as e:
            error_message = (f"⚠️ Could not connect to Ollama.\n\n '{e}'")
            answer_placeholder.error(error_message)

        except Exception as e:
            error_message = (f"⚠️ Ollama returned an error.\n\n'{e}'")
            answer_placeholder.error(error_message)

# add buttons to stop chatting or save the chat
col1, col2 = st.columns(2)
with col1:
    new_chat = st.button(
        "Start New Chat",
        width="stretch",
        icon="🗯️",
        help="this button will reload the page and delete all messages," \
        "SAVE your chat first, or you'll lose these messages!")
with col2:
    save_chat = st.button(
        "Save Chat",
        width="stretch",
        icon="💾",
        help="remember to give an appropriate name to the chat!")
    
if new_chat:
    st.session_state.chat_start = True
    st.session_state.chat_active = True
    st.session_state.chat_messages = []
    st.session_state.chat_started_at = None
    st.rerun()

if save_chat:
    if not st.session_state.chat_active:
        st.warning("There is no active chat to save.")
    elif not st.session_state.chat_messages:
        st.warning("The current chat is empty.")
    else:
        os.makedirs(CHAT_FOLDER, exist_ok=True)
        chat_filepath = os.path.join(CHAT_FOLDER, f"{CHAT_NAME}.md")
        saved_file = functions.save_chat_as_markdown(
            st.session_state.chat_messages,
            st.session_state.chat_started_at,
            chat_filepath
        )
                # Save structured data as well as the human-readable Markdown export.
        # JSON preserves each message's role and the assistant's reasoning.
        chat_data = {
            "started_at": st.session_state.chat_started_at.isoformat()
                if st.session_state.chat_started_at else None,
            "model": st.session_state.get("chat_model", MODEL),
            "system_prompt": st.session_state.get("chat_system_prompt", ""),
            "messages": st.session_state.chat_messages,
        }

        with open(chat_json_path(CHAT_NAME), "w", encoding="utf-8") as f:
            json.dump(chat_data, f, ensure_ascii=False, indent=2)

        st.toast("Chat saved!", icon="✅")

