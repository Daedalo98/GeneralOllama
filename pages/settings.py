import streamlit as st
import os
import json
import functions
import requests

# =========================================================
# OLLAMA
# =========================================================
st.markdown("# Ollama settings")

CONTEXT_LENGTHS = {
    "1K": 1_024,
    "2K": 2_048,
    "4K": 4_096,
    "8K": 8_192,
    "16K": 16_384,
    "24K": 24_576,
    "32K": 32_768,
    "48K": 49_152,
    "64K": 65_536,
    "80K": 81_920,
    "96K": 98_304,
    "112K": 114_688,
    "128K": 131_072,
}

with st.container(border=False):
    with st.expander("Ollama settings", expanded=True):
        # OLLAMA_HOST = "http://127.0.0.1:11434"
        OLLAMA_HOST = st.text_input("Ollama main URL", value="http://127.0.0.1:11434")
        # share Ollama host with all other pages
        st.session_state.ollama_host = OLLAMA_HOST  

        # initialize models list
        models = []

        # Determine current state
        process = st.session_state.ollama_process

        # Did our subprocess die?
        if process is not None and process.poll() is not None:
            st.session_state.ollama_process = None
            st.session_state.ollama_started_by_app = False
            process = None

        selected_context_name = st.selectbox(
            "Context length",
            options=list(CONTEXT_LENGTHS.keys()),
            index=list(CONTEXT_LENGTHS.values()).index(st.session_state.context_length),
            disabled=st.session_state.ollama_process != None,
        )

        selected_context = CONTEXT_LENGTHS[selected_context_name]
        st.caption(f"Ollama context: `{selected_context:,}` tokens")
        # share ollama context lenght with other pages of the app

        ollama_running = functions.ollama_is_running(OLLAMA_HOST=OLLAMA_HOST, verbose=False)
        if ollama_running:
            if st.session_state.ollama_started_by_app:
                st.success("Ollama is running (started by this app)")
            else:
                st.success("Ollama is running")
                st.warning("if you want to stop Ollama, do it from outside the app. use BASH command: sudo killall -s 9 ollama")
                        
            # Add Ollama Models
            try:
                response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=2)
                if response.status_code == 200:
                    data = response.json()
                    for m in data.get("models", []):
                        models.append(m["name"])
                    # share models with all pages of the app
                    st.session_state.ollama_models = models
            
            except Exception as e:
                st.warning(f"Error fetching Ollama models: {e}")
                pass # fail if Ollama is not running
        else:
            st.warning("Ollama is not running. Please start Ollama first.")

            if st.button("▶ Start Ollama", width="stretch", key="start_ollama"):
                try:
                    process = functions.start_ollama(
                        OLLAMA_HOST=OLLAMA_HOST,
                        OLLAMA_CONTEXT_LENGTH=selected_context)
                    
                    st.session_state.ollama_process = process
                    st.session_state.ollama_started_by_app = True
                    st.session_state.context_length = selected_context
                    st.success(f"Started Ollama with {selected_context:,} token context.")

                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to start Ollama: {e}")

        # Stop button
        if st.session_state.ollama_started_by_app:
            if st.button("⏹ Stop Ollama", width="stretch", key="stop_ollama"):
                try:
                    functions.stop_ollama(OLLAMA_HOST, st.session_state.ollama_process)
                except Exception as e:
                    st.warning(f"failed to stop Ollama with error: {e}")
                finally:
                    st.session_state.ollama_process = None
                    st.session_state.ollama_started_by_app = False

                    st.rerun()


# =========================================================
# PROMPTS
# =========================================================
st.markdown("# Prompts management")

PROMPTS_FILE = "./prompts.json"
if not os.path.exists(PROMPTS_FILE):
    # Create a default prompts.json if it doesn't exist
    default_prompts = {
        "OCR_PROMPT": """
        Extract the content of this document image into clean Markdown.

        Rules:
        1. Extract all visible text and preserve reading order. 
        2. Preserve paragraphs, headings and subheadings using Markdown headings appropriately.
        3. If there are tables, reproduce them as Markdown tables preserving headers and rows.
        """.strip(),

        "CHAT_PROMPT": """
        You are a helpful assistant.

        Rules:
        1. Respond to the user's query accurately and helpfully.
        2. If you don't know the answer, say so.
        3. Keep your responses concise but informative.
        """.strip()
    }
    with open(PROMPTS_FILE, "w") as f:
        json.dump(default_prompts, f, indent=4)

# add prompt dictionary in state to share them with all pages
st.session_state.prompts_dict = functions.load_prompts(PROMPTS_FILE)
prompt_names = list(st.session_state.prompts_dict.keys())

# Optional safety check
if not prompt_names:
    st.error("No prompts were loaded from prompts.json.")
    st.stop()

with st.container(border=False):
    with st.expander("Prompt Management", expanded=False):
        st.write("You can manage your prompts here. Add new prompts, edit existing ones, or delete them.")
        prompts_dict = functions.load_prompts(PROMPTS_FILE)

        # Display existing prompts
        for prompt_name, prompt_text in prompts_dict.items():
            with st.container():
                st.markdown("---")
                prompt_text = st.text_area(f"{prompt_name}", value=prompt_text, key=f"edit_{prompt_name}", height="content")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Save modified prompt", key=f"save_{prompt_name}", width="stretch"):
                        functions.save_prompt(PROMPTS_FILE, prompt_name, prompt_text)
                        st.rerun()

                with col2:
                    if st.button(f"Delete {prompt_name}", key=f"delete_{prompt_name}", type="primary", width="stretch"):
                        del prompts_dict[prompt_name]
                        functions.save_prompt(PROMPTS_FILE, prompts_dict, prompt_text)
                        st.rerun()

        # Add new prompt
        new_prompt_name = st.text_input("New Prompt Name", key="new_prompt_name")
        new_prompt_text = st.text_area("New Prompt Text", key="new_prompt_text")
        if st.button("Add New Prompt"):
            if new_prompt_name and new_prompt_text:
                prompts_dict[new_prompt_name] = new_prompt_text
                functions.save_prompt(PROMPTS_FILE, new_prompt_name, new_prompt_text)
                st.rerun()
            else:
                st.warning("Please provide both a name and text for the new prompt.")
                