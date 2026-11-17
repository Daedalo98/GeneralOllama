import streamlit as st    
from io import StringIO
import functions
import requests
import json


prompt_names = list(st.session_state.prompts_dict.keys())
ollama_models = st.session_state.ollama_models
OLLAMA_HOST = st.session_state.ollama_host

if "is_quizzing" not in st.session_state:
    st.session_state.is_quizzing = False


# Store quiz data indexed by filename (str), NOT the UploadedFile object.
# Structure:
# {
#   "notes.md": {
#       "source_text": "...",
#       "items": [
#           {"question": "...", "user_answer": "", "graded_answer": None, "feedback": None},
#           ...
#       ]
#   }
# }
if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = {}


# ---------------------------------------------------------
# QUIZ TIME
# ---------------------------------------------------------
with st.sidebar:
    with st.container(border=False):
        with st.expander("❓ Create QUIZ", expanded=True):

            uploaded_files = st.file_uploader(
                "Choose a .md file",
                type=["md", "json"],
                accept_multiple_files=True,
                disabled=st.session_state.is_parsing,
                help="Upload a markdown file you want to work with."
            )

            N_questions = st.number_input("How many questions per document?", min_value=1, step=1, value=5)
    
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
                        prompt_names.index("QUIZ_questions")
                        if "QUIZ_questions" in prompt_names else 0)
    
                    prompt_name = st.selectbox(
                        "Active Prompt",
                        options=prompt_names,
                        index=default_index,
                        key="quiz_prompt_selector"
                    )
    
                    quiz_prompt_text = st.session_state.prompts_dict.get(prompt_name,"")
                    # show the prompt
                    st.text(quiz_prompt_text)

            if st.button("Start QUIZ", 
                         #disabled=st.session_state.is_parsing, 
                         type="primary", 
                         width="stretch",
                         help="re-starting the quiz will delete previous questions."):
                st.session_state.is_quizzing = True

st.markdown("# Generate questions from your notes")
if not uploaded_files:
    st.markdown("start uploading the notes you want to review in the sidebar.")

MAX_RETRIES = 3
if st.session_state.is_quizzing:
    for i, uploaded_file in enumerate(uploaded_files):
        doc_name = uploaded_file.name
        string_data = uploaded_file.getvalue().decode("utf-8")
        st.session_state.quiz_data[doc_name] = {"source_text": string_data, "items": []}

        with st.container(border=False, key=f"quiz_cont_{i}", width="stretch"):
            st.info(f"Generating questions regarding {doc_name}...")

            llm_payload = (
                f"--- START SOURCE ---\n"
                f"{string_data}\n"
                f"--- END SOURCE ---\n"
                f"Number of questions needed: {N_questions}.\n"
                f"Return ONLY a valid JSON array. Each element must be an object containing exactly one field:\n"
                f"'question', whose value is a nonempty string.\n"
                f"Do not include Markdown code fences, introductory text, or commentary.\n"
                f"Generate exactly the requested number of questions whenever possible.\n"
                f"Do not answer the generated questions.\n"
            )

            ollama_messages =[     {
                    "role": "system",
                    "content": quiz_prompt_text
                },      {
                    "role": "user",
                    "content": llm_payload
                }       ]

            raw_response = ""
            for attempt in range(MAX_RETRIES + 1):
                raw_response = ""  # <-- FIX: Reset accumulator on every retry!

                # Reasoning display
                st.session_state.thinking_text = ""
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
                            raw_response += chunk

                        elif event_type == "done":
                            break

                except requests.exceptions.Timeout:
                    error_message = ("⚠️ The request to Ollama timed out.")
                    st.warning(error_message)

                except requests.exceptions.RequestException as e:
                    error_message = (f"⚠️ Could not connect to Ollama.\n\n`{e}`")
                    st.warning(error_message)

                except Exception as e:
                    error_message = (f"⚠️ Ollama returned an error.\n\n`{e}`")
                    st.warning(error_message)

                # Validate the response
                validated = functions.validate_questions(raw_response)
                if validated is not None:
                    break

            try:
                data = json.loads(raw_response)
                # Save both the source context and structured question objects
                st.session_state.quiz_data[doc_name] = {
                    "source_text": string_data,
                    "items": [
                    {
                        "question": item["question"],
                        "user_answer": "",
                        "graded_answer": None,  # Tracks the exact answer text last evaluated
                        "feedback": None,
                    }
                    for item in data
                    ],
                }
            except Exception as exc:  # <-- FIX: Catch Exception, not a string variable
                st.warning(f"Could not parse questions for {doc_name}: {exc}")

        # Generation is complete; exit generation mode so reruns don't regenerate
        st.session_state.is_quizzing = False


# =========================================================
# PHASE 2: INTERACTIVE QUIZ WORKSPACE (Runs on every rerun)
# =========================================================
if st.session_state.quiz_data:
    for doc_name, doc_bundle in st.session_state.quiz_data.items():
        st.subheader(f"📄 {doc_name}")
        source_text = doc_bundle["source_text"]

        for idx, item in enumerate(doc_bundle["items"], start=1):
            # Unique prefix prevents Streamlit DuplicateWidgetID errors across files/questions
            widget_prefix = f"{doc_name}_q_{idx}"

            with st.container(border=True):
                st.markdown(f"**Question {idx}:** {item['question']}")

                # 1. Space for the user to write their own answer
                user_input = st.text_area(
                    label="Your answer:",
                    value=item["user_answer"],
                    key=f"{widget_prefix}_input",
                    height=120,
                    placeholder="Type your answer in your own words...",
                )
                # Keep session state synced with what the user typed
                item["user_answer"] = user_input

                # 2. Grade the answer when the user clicks the button
                if st.button(
                    "Check Answer",
                    key=f"{widget_prefix}_btn",
                    disabled=not user_input.strip(),  # Prevent empty submissions
                ):
                    with st.spinner("Evaluating your answer against the source notes..."):
                        try:
                            feedback = functions.evaluate_user_answer(
                                source_text=source_text,
                                question=item["question"],
                                user_answer=user_input,
                                model=MODEL,
                                ollama_url=OLLAMA_URL,
                                temperature=0.2,  # Lower temp = more consistent grading
                                max_tokens=MAX_TOKENS,
                                timeout=REQUEST_TIMEOUT,
                            )
                            item["feedback"] = feedback
                            item["graded_answer"] = user_input.strip()  # <-- Mark this version as graded

                        except requests.exceptions.RequestException as e:
                            st.error(f"Could not reach Ollama: {e}")

                # 3. Persistently display feedback once generated
                if item["feedback"]:
                    st.markdown("---")
                    st.markdown("**Tutor Feedback:**")
                    st.info(item["feedback"])
                    


        # =========================================================
        # BATCH GRADING: "Check All Answers" for this document
        # Placed AFTER the question loop so item["user_answer"]
        # already holds the latest text from every st.text_area.
        # =========================================================

        # Select ONLY questions that have a non-empty answer AND haven't been graded in their current form
        pending_items = [
            (idx, item)
            for idx, item in enumerate(doc_bundle["items"], start=1)
            if item["user_answer"].strip()
            and item["user_answer"].strip() != item.get("graded_answer")
        ]

        if st.button(
            f"Check Remaining Answers for {doc_name} ({len(pending_items)} pending)",
            key=f"{doc_name}_check_all_btn",
            type="primary",
            disabled=len(pending_items) == 0,  # Disabled if everything typed is already graded
            use_container_width=True,
        ):
            progress_bar = st.progress(0, text="Starting batch evaluation...")
            total_to_grade = len(pending_items)

            for step, (idx, item) in enumerate(pending_items, start=1):
                progress_bar.progress(
                    int(((step - 1) / total_to_grade) * 100),
                    text=f"Evaluating Question {idx} ({step} of {total_to_grade})...",
                )

                try:
                    feedback = functions.evaluate_user_answer(
                        source_text=source_text,
                        question=item["question"],
                        user_answer=item["user_answer"],
                        model=MODEL,
                        ollama_url=OLLAMA_URL,
                        temperature=0.2,
                        max_tokens=MAX_TOKENS,
                        timeout=REQUEST_TIMEOUT,
                    )
                    item["feedback"] = feedback
                    item["graded_answer"] = item["user_answer"].strip()  # <-- Mark as graded

                except requests.exceptions.RequestException as e:
                    st.error(f"Error evaluating Question {idx}: {e}")
                    break

            progress_bar.progress(100, text="All pending answers evaluated!")
            st.rerun()