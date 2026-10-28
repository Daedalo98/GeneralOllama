import streamlit as st
from streamlit_player import st_player
import functions
import requests
import json
import os
from pathlib import Path

prompt_names = list(st.session_state.prompts_dict.keys())
ollama_models = st.session_state.ollama_models
OLLAMA_HOST = st.session_state.ollama_host
VIDEO_FOLDER = st.session_state.folder_paths["VIDEO_FOLDER"]

# ==========================================
# STATES
# ==========================================
if "metadata" not in st.session_state:
    st.session_state.metadata = {}
if "transcript" not in st.session_state:
    st.session_state.transcript = []
if "video_url" not in st.session_state:
    st.session_state.video_url = ""
if "video_id" not in st.session_state:
    st.session_state.video_id = ""
if "start_time" not in st.session_state:
    st.session_state.start_time = 0.0

if "enhanced_yt" not in st.session_state:
    st.session_state.enhanced_yt = ""

with st.sidebar:
    # ---------------------------------------------------------
    # YOUTUBE TRANSCRIPT SETTINGS
    # ---------------------------------------------------------
    with st.container(border=False):
        with st.expander("🎬 YouTube", expanded=True):
            # option 1: fetch new video
            st.subheader("Ingest New Video")
            with st.form("fetch_transcript_form"):
                input_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
                submit_btn = st.form_submit_button("Fetch & Process Transcript", type="primary")
        
                if submit_btn:
                    st.session_state.render_yt_ui = True
                    if input_url.strip():  
                        video_id = functions.extract_video_id(input_url)
                        functions.add_log(f"User requested extraction for URL: {input_url}")
                        
                        # --- PURGE OLD STATE ---
                        st.session_state.transcript = []
                        st.session_state.metadata = {}
                        st.session_state.video_url = ""
                        st.session_state.video_id = ""
                        st.session_state.enhanced_yt = ""
                        #st.session_state.quiz_state = "setup"

                        for key in list(st.session_state.keys()):
                            if key.startswith("text_") or key.startswith("btn_"):
                                del st.session_state[key]
                        
                        # --- FETCH NEW DATA ---
                        st.session_state.video_url = input_url
                        st.session_state.video_id = video_id
                        
                        with st.status(f"Processing Request for {video_id}...", expanded=True) as status:
                            # Removed CACHE_DIR argument!
                            full_data_dict = functions.fetch_transcript_with_logs(input_url, video_id)
                            
                            if full_data_dict and full_data_dict.get("segments"):
                                st.session_state.transcript = full_data_dict["segments"]
                                st.session_state.metadata = full_data_dict.get("metadata", {}) 
                                st.session_state.start_time = 0.0
                                
                                functions.add_log(f"Successfully loaded transcript for {video_id}.")
                                status.update(label="Transcript Ready!", state="complete", expanded=False)
                                
                                # ONLY RERUN IF SUCCESSFUL!
                                st.rerun() 
                            else:
                                functions.add_log(f"Failed to fetch data for {video_id}.")
                                status.update(label="Failed to fetch valid data. Read logs above.", state="error", expanded=True)
                                # NO RERUN HERE. Let the user read the error message.
                    else:
                        st.warning("Please enter a valid YouTube URL first.")

            st.divider()

            # option 2: load from file
            st.subheader("Load Saved Workspace")
            uploaded_file = st.file_uploader("Upload a previously downloaded .json file", type=["json"])

            if uploaded_file is not None:
                if st.button("📁 Load Data from File", width="stretch"):
                    try:
                        # Read the JSON file the user just uploaded
                        loaded_data = json.load(uploaded_file)
                        
                        # Populate session state
                        st.session_state.transcript = loaded_data.get("segments", [])
                        st.session_state.metadata = loaded_data.get("metadata", {})
                        st.session_state.video_id = loaded_data.get("video_id", "")
                        st.session_state.video_url = loaded_data.get("metadata", {}).get("video_url", "")
                        
                        st.toast("Data successfully loaded from file!", icon="✅")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to load file. It might be corrupted. Error: {e}")

            # VIDEO PLAYBACK (Shared by both logic paths)
            if st.session_state.video_id and st.session_state.transcript:
                st.divider()
                st.subheader("Video Playback")
                
                nocookie_url = f"https://www.youtube-nocookie.com/embed/{st.session_state.video_id}"
                
                st_player(
                    nocookie_url,
                    config={"playerVars": {"start": int(float(st.session_state.start_time)), "autoplay": 0}},
                    key=f"player_{st.session_state.start_time}"
                )

            # Export Workspace
            st.subheader("💾 Export Current Workspace")
            # Check if there is actual data to export
            if st.session_state.get("transcript"):
                
                # Bundle the data they need to restore the session
                export_dict = {
                    "video_id": st.session_state.video_id,
                    "metadata": st.session_state.metadata,
                    "segments": st.session_state.transcript,
                    "enhanced_text": st.session_state.enhanced_yt
                }
                
                # Convert dict to a formatted JSON string
                json_string = json.dumps(export_dict, indent=4)
                
                st.download_button(
                    label="📥 Download Session as JSON",
                    data=json_string,
                    file_name=f"transcript_data_{st.session_state.video_id}.json",
                    mime="application/json",
                    width="stretch"
                )
            else:
                st.info("No active video to export.")

    # ---------------------------------------------------------
    # Note Generation settings
    # ---------------------------------------------------------
    with st.container(border=False):             
        with st.expander("🧠 Generative model settings", expanded=False):
            # model selection
            text_idx = ollama_models.index("granite4.2:3b") if "granite4.2:3b" in ollama_models else 0
            MODEL = st.selectbox("Main Model", ollama_models, index=text_idx)

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

                st.subheader("System Prompts")
                default_index = (
                    prompt_names.index("GENERATE_MD_from_YT")
                    if "GENERATE_MD_from_YT" in prompt_names else 0)

                prompt_name = st.selectbox(
                    "Active Prompt",
                    options=prompt_names,
                    index=default_index,
                    key="video_prompt_selector"
                )

                video_prompt_text = st.session_state.prompts_dict.get(prompt_name,"")
                # show the prompt
                st.text(video_prompt_text)

            # path to save the chat
            VIDEO_NAME = st.text_input(
                "Name the video note (without extension) before saving it!", 
                value=f"{st.session_state.video_id}", 
                help=f"by default, chats will be stored as .md files in {VIDEO_FOLDER}")
            

# =========================================================
# YOUTUBE SECTION OF THE MAIN APP
# =========================================================
# Display Video Info
if "metadata" in st.session_state and st.session_state.metadata and st.session_state.render_yt_ui:
    with st.container(height=700, width="stretch", border=False):
        st.info(f"**Diagnostic Check:** Current Video ID in memory is `{st.session_state.video_id}`")
        meta = st.session_state.metadata
        st.markdown(f"### 📺 {meta.get('title', 'Unknown Title')}")
        st.caption(f"**Channel:** {meta.get('author_name', 'Unknown')} | **Uploaded:** {meta.get('upload_date', 'Unknown')} | **Source:** [Link]({meta.get('video_url', '')})")
        st.divider()

        for i, segment in enumerate(st.session_state.transcript):
            # Failsafe: Ensure segment is actually a dictionary
            if not isinstance(segment, dict):
                continue
                
            start_sec = segment.get('start', 0.0)
            timestamp_str = functions.format_timestamp(start_sec)
            current_text = str(segment.get('text', '')) # Force as string
            
            text_length = len(current_text)
            btn_ratio = 0.15
            text_ratio = 1 - btn_ratio
            btn_col, text_col = st.columns([btn_ratio, text_ratio])
            
            # 🚨 THE FIX: Create globally unique keys tied to THIS specific video
            unique_btn_key = f"btn_{st.session_state.video_id}_{i}"
            unique_txt_key = f"text_{st.session_state.video_id}_{i}"
            
            with btn_col:
                if st.button(f"⏱️ {timestamp_str}", key=unique_btn_key, width='stretch'):
                    st.session_state.start_time = start_sec
                    st.rerun()
            
            with text_col:
                # 🚨 THE FIX: The callback must grab the exact unique key
                def update_text(index=i, key_name=unique_txt_key):
                    # Update the main memory with the new text
                    st.session_state.transcript[index]['text'] = st.session_state[key_name]
                    st.toast("💾 Auto-saved!", icon="✅")

                st.text_area(
                    "Edit Text", 
                    value=current_text, 
                    key=unique_txt_key,  # Applies the unique key here!
                    label_visibility="collapsed",
                    on_change=update_text,
                    height="content"
                )
            
            st.markdown("<hr style='margin: 0.2em 0px; border-top: 1px dashed #ddd;'>", unsafe_allow_html=True)


    with st.container(border=False):
        if st.button("Create MD note from the transcription", type="primary", width="stretch", icon="🧠"):
            meta = st.session_state.metadata
            # Extract just the text from the segment dictionaries
            text_only = "\n".join([seg.get("text", "") for seg in st.session_state.transcript])

            llm_payload = (
                f"--- METADATA ---\n"
                f"Title: {meta.get('title', 'Unknown')}\n"
                f"Author: {meta.get('author_name', 'Unknown')}\n"
                #f"Date: {meta.get('upload_date', 'Unknown')}\n"
                #f"URL: {meta.get('video_url', 'Unknown')}\n\n"
                f"--- REDACTED TRANSCRIPT ---\n"
                f"{text_only}"
            )

            ollama_messages =[     {
                    "role": "system",
                    "content": st.session_state.chat_system_prompt
                },      {
                    "role": "user",
                    "content": llm_payload
                }       ]

            st.session_state.enhanced_yt = (
                f"--- METADATA ---\n"
                f"Title: {meta.get('title', 'Unknown')}\n"
                f"Author: {meta.get('author_name', 'Unknown')}\n"
                f"Date: {meta.get('upload_date', 'Unknown')}\n"
                f"URL: {meta.get('video_url', 'Unknown')}\n"
                f"--- METADATA ---\n\n"
            )
            st.session_state.thinking_text = ""

            # Reasoning display
            thinking_expander = st.expander("🧠 Reasoning", expanded=False)
            with thinking_expander:
                thinking_placeholder = st.empty()

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
                    thinking=True # Enable Ollama thinking.
                    ):

                    event_type = event["type"]
                    chunk = event.get("content", "")

                    #print(f"EVENT TYPE: {event_type} | CONTENT: {chunk[:100]!r}")

                    # THINKING
                    if event_type == "thinking":
                        st.session_state.thinking_text += chunk

                        # Update the reasoning area.
                        with thinking_expander:
                            thinking_placeholder.markdown(st.session_state.thinking_text)                        

                    # FINAL ANSWER
                    elif event_type == "content":
                        st.session_state.enhanced_yt += chunk

                    elif event_type == "done":
                        break

                # Render final answer
                # answer_placeholder.markdown(st.session_state.enhanced_yt)

            except requests.exceptions.Timeout:
                error_message = ("⚠️ The request to Ollama timed out.")
                st.warning(error_message)

            except requests.exceptions.RequestException as e:
                error_message = (f"⚠️ Could not connect to Ollama.\n\n`{e}`")
                st.warning(error_message)

            except Exception as e:
                error_message = (f"⚠️ Ollama returned an error.\n\n`{e}`")
                st.warning(error_message)

        # save or edit .md note
        col1, col2 = st.columns(2)
        with col1:
            edit_enhanced_yt = st.button("Edit / Preview", key="YT enhanced text E/P", width="stretch") 

        # editable answer
        if edit_enhanced_yt:
            st.session_state.edit_text = not st.session_state.edit_text
            st.rerun()

        if st.session_state.edit_text:
            st.session_state.enhanced_yt = st.text_area(
                "Markdown text",
                value=st.session_state.enhanced_yt,
                height="content",
                key="markdown_editor",
            )
        else:
            st.markdown(st.session_state.enhanced_yt)

        with col2:
            if st.button("Download notes as .md file", width="stretch"):

                # Specify the output path
                file_path = os.path.join(VIDEO_FOLDER, f"{VIDEO_NAME}.md")
                file_path = Path(file_path)
                
                # Save the Markdown file
                file_path.write_text(st.session_state.enhanced_yt, encoding="utf-8")
                st.toast("Note downloaded", duration="short", icon="✅")