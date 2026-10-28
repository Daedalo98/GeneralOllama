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

# Question records indexed by source document.
if "questions" not in st.session_state:
    st.session_state.questions = {}

# Track the currently active quiz.
if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = {
        "type": None,
        "question_ids": [],
    }


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
    for i, filepath in enumerate(uploaded_files):
        st.session_state.questions[filepath] = ""
        with st.container(border=False, key=f"quiz_cont_{i}", width="stretch"):
            # To convert to a string based IO:
            stringio = StringIO(filepath.getvalue().decode("utf-8"))
            #st.write(stringio)

            # To read file as string:
            string_data = stringio.read()
            #st.write(string_data)

            st.info(f"Questions regarding {filepath.name}:")

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

            for attempt in range(MAX_RETRIES + 1):

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
                            st.session_state.questions[filepath] += chunk

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

                # Accept the response only if validation succeeds.
                questions = functions.validate_questions(st.session_state.questions[filepath])

                # Stop retrying once a valid response is obtained.
                if questions is not None:
                    break

            try:
                # Parse it into a Python list of dicts
                data = json.loads(st.session_state.questions[filepath])
                st.session_state.questions[filepath] = []
                            
                # Iterate, extracting the "question" field for each item
                for idx, item in enumerate(data, start=1):
                    question = item["question"]
                    # At this point you have the full question text
                    # print(f"Question {idx}: {question}")
                    # Or, inside a Streamlit app:
                    st.write(f"Question {idx}: {question}")
                    st.session_state.questions[filepath].append(question)

            except error_message:
                st.warning("the answer was not well formatted, try again.")

    st.session_state.is_quizzing = False # exist "is_quizzing" state when all answers have been produced.

    # Display the editable table.
    edited_df = functions.render_question_table(
        questions,
        editor_key="question_bank_editor",
    )

    # Save changes only when explicitly requested.
    if st.button("Save changes"):

        # Synchronize the edits with the question records.
        questions = functions.apply_table_edits(questions, edited_df)

        # Persist the updated question bank.
        functions.save_question_bank(bank_path, questions)

        st.success("Question bank saved.")



# Illustrative workflow for a single question.

question = selected_question["question"]

# Generate exactly one answer for this question.
answer = generate_answer_with_ollama(
    question=question,
    source_text=source_text,
    answer_prompt=answer_prompt,
)

# Save the generated answer in the corresponding record.
selected_question["answer"] = answer

