import streamlit as st
import os
import re
import time
import json
import functions
import requests

# --- imports for the VectorDB ---
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

prompt_names = list(st.session_state.prompts_dict.keys())
ollama_models = st.session_state.ollama_models
OLLAMA_HOST = st.session_state.ollama_host
MD_FOLDER = st.session_state.folder_paths["MD_FOLDER"]
DB_PATH = st.session_state.folder_paths["DB_FOLDER"]

# =========================================================
# RAG CHAT SESSION    
# =========================================================
if "rag_config" not in st.session_state:
    st.session_state.rag_config = {}

if "query_text" not in st.session_state:
    st.session_state.query_text = ""

if "enhanced_query" not in st.session_state:
    st.session_state.enhanced_query = False

if "query_editing" not in st.session_state:
    st.session_state.query_editing = False

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

with st.sidebar:
    with st.container(border=False):   
        with st.expander("📚 RAG settings", expanded=True):
            embedding_idx = ollama_models.index("nomic-embed-text-v2-moe:latest") if "nomic-embed-text-v2-moe:latest" in ollama_models else 0
            embedding_model = st.selectbox("Embedding Model", ollama_models, index=embedding_idx)
            st.session_state.rag_config['embedding_model'] = embedding_model
            
            # EMBEDDING & CHUNKING PARAMETERS
            with st.expander("Embedding & Chunking Parameters", expanded=False):
                chunk_strategy = st.selectbox("Chunking Strategy", ["Recursive Character", "Token-Based", "Strict Character"])
                st.session_state.rag_config['chunk_strategy'] = chunk_strategy
                chunk_size = st.number_input("Chunk Size", min_value=100, max_value=8000, value=1300, step=100)
                st.session_state.rag_config['chunk_size'] = chunk_size
                chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=2000, value=350, step=50)
                st.session_state.rag_config['chunk_overlap'] = chunk_overlap
            
                if chunk_strategy in ["Recursive Character", "Strict Character"]:
                    # Let user define exact separators, defaulting to standard Markdown structural breaks
                    raw_separators = st.text_input("Separators (comma separated list, highest to lowest priority):", value="\n\n, \n, . ,  ,#, ##,###,")
                    # Parse the string into actual escape characters
                    separators = [sep.strip().replace('\\n', '\n') for sep in raw_separators.split(",")]
                else:
                    st.info("Separators are intrinsically handled by the tokenizer for Token-Based splitting.")
                    separators = None
                st.session_state.rag_config['separators'] = separators

                embedding_dimensions = st.number_input("Embedding Dimensions (if configurable)", min_value=128, max_value=8192, value=1536)
                st.session_state.rag_config['embedding_dimensions'] = embedding_dimensions
                distance_metric = st.selectbox("Distance Metric", ["Cosine Similarity", "L2 (Euclidean)", "Inner Product"])
                st.session_state.rag_config['distance_metric'] = distance_metric

            # DATABASE PERSISTENCE & INITIALIZATION (Place this BEFORE the button)
            # Automatically load the database if it exists on disk, so it's visible on future loads
            if "vector_db" not in st.session_state:
                if len(os.listdir(DB_PATH)) > 1:
                    st.info("📦 Existing VectorDB found in this folder.")
                    try:
                        embeddings = OllamaEmbeddings(model=st.session_state.rag_config.get('embedding_model'))
                        st.session_state.vector_db = Chroma(
                            persist_directory=DB_PATH, 
                            embedding_function=embeddings)
                        st.success("🔄 Existing VectorDB successfully reconnected on load!")
                    except Exception as e:
                        st.sidebar.error(f"Could not auto-load existing DB: {e}")
                    
                else:
                    st.info("✨ No existing VectorDB found. A new one will be created.")
                    st.session_state.vector_db = None

            # Display DB status in the UI
            if st.session_state.vector_db is not None:
                st.info(f"🟢 Active Database Loaded: `{DB_PATH}`")
            else:
                st.warning("🔴 No active database loaded in this session yet.")

            
            # DATABASE CREATION BUTTON
            if st.button("Save Configuration and Execute Database Creation", help="only adds new files to the db. if you want to wipe the db and rebuild it, manually delete the folder."):
                st.session_state.rag_config.update({
                    'chunk_strategy': chunk_strategy,
                    'chunk_size': chunk_size,
                    'chunk_overlap': chunk_overlap,
                    'separators': separators,
                    'embedding_model': embedding_model,
                    'embedding_dimensions': embedding_dimensions,
                    'distance_metric': distance_metric
                })
            
                config = st.session_state.rag_config
                with st.spinner("Preparing pipeline..."):
                    try:
                        embeddings = OllamaEmbeddings(model=config['embedding_model'])
                        metric_mapping = {"Cosine Similarity": "cosine", "L2 (Euclidean)": "l2", "Inner Product": "ip"}
                        collection_metadata = {"hnsw:space": metric_mapping.get(config['distance_metric'])}

                        # Initialize / Create the DB client
                        vector_db = Chroma(
                            persist_directory=DB_PATH, 
                            embedding_function=embeddings,
                            collection_metadata=collection_metadata
                        )

                        # Load Markdown Files (Using the exclude fix from earlier)
                        st.toast("Scanning for Markdown files...")
                        loader = DirectoryLoader(MD_FOLDER, glob="**/*.md", exclude=["**/chroma_db/**"], loader_cls=TextLoader)
                        all_docs = loader.load()
                        
                        if not all_docs:
                            st.warning(f"No .md files found in {MD_FOLDER}.")
                            time.sleep(1)
                            st.rerun()

                        # Smart Append Filter
                        docs_to_process = all_docs
                        try:
                            existing_data = vector_db.get(include=["metadatas"])
                            if existing_data and existing_data['metadatas']:
                                existing_sources = set(meta.get('source') for meta in existing_data['metadatas'])
                                docs_to_process = [d for d in all_docs if d.metadata.get('source') not in existing_sources]
                        except Exception as e:
                            st.warning(f"Could not read existing DB metadata: {e}")

                        if not docs_to_process:
                            st.success("✅ Database is already up to date!")
                            st.session_state.vector_db = vector_db
                            time.sleep(1)
                            st.rerun()

                        # Configure Splitter
                        if config['chunk_strategy'] == "Recursive Character":
                            splitter = RecursiveCharacterTextSplitter(chunk_size=config['chunk_size'], chunk_overlap=config['chunk_overlap'], separators=config['separators'])
                        elif config['chunk_strategy'] == "Strict Character":
                            splitter = CharacterTextSplitter(chunk_size=config['chunk_size'], chunk_overlap=config['chunk_overlap'], separator=config['separators'][0])
                        else: 
                            splitter = TokenTextSplitter(chunk_size=config['chunk_size'], chunk_overlap=config['chunk_overlap'])

                        # Batching process
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i, doc in enumerate(docs_to_process):
                            chunks = splitter.split_documents([doc])
                            if chunks:
                                vector_db.add_documents(chunks)
                            progress_bar.progress((i + 1) / len(docs_to_process))
                        
                        # CRITICAL: Save back to session state so it survives the upcoming rerun
                        st.session_state.vector_db = vector_db
                        st.success("✅ VectorDB loaded and saved to session!")
                        time.sleep(1)
                        st.rerun() # Force rerun to refresh the top-level UI indicators

                    except Exception as e:
                        st.error(f"❌ An error occurred: {str(e)}")

            with st.expander("RAG parameters", expanded=False):
                top_k = st.number_input("Initial Retrieval Top-K", min_value=1, max_value=50, value=10, help="How many raw chunks to pull from the DB.")
                sim_threshold = st.slider("Similarity Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05, help="Minimum relevance score (0-1) to be considered.")
                alpha_weight = st.slider("Hybrid Alpha (α)", min_value=0.0, max_value=1.0, value=0.5, step=0.1, help="0.0 = Pure Keyword | 1.0 = Pure Semantic.")

                use_hyde = st.toggle("Enable HyDE", value=False, help="Hypothetical Document Embeddings.")
                use_step_back = st.toggle("Enable Step-Back", value=False, help="Prompts the LLM to generate a broader version of the query.")

                with st.expander("Model Settings", expanded=False, icon="⚙️"):
                    text_idx = ollama_models.index("granite4.2:3b") if "granite4.2:3b" in ollama_models else 0
                    MODEL = st.selectbox("Main Model", ollama_models, index=text_idx, key="RAG model")

                    # Ollama's local HTTP API.
                    OLLAMA_URL = st.text_input("Ollama URL", value=f"{OLLAMA_HOST}/api/chat")
    
                    # Generation temperature.
                    # CHAT should generally be deterministic, so 0 is appropriate.
                    TEMPERATURE = st.number_input("Temperature", 0.0, 3.0, 0.6, 0.1, help="Higher values make output more random. For CHAT, 0 is recommended for deterministic output.")
    
                    # Maximum number of generated tokens.
                    MAX_TOKENS = st.number_input("Max Tokens", -1, 100000, 10000, 1, help="Maximum number of tokens to generate. -1 = no limit.")
    
                    # HTTP timeout.
                    REQUEST_TIMEOUT = st.number_input("Request Timeout", 1, 1000, 300, 1, help="Timeout for CHAT requests in seconds.")
    
                    STREAMING = st.toggle("Streaming Generation", value=True, key="chat_streaming", help="Whether to stream the model's output as it generates. If disabled, the model will return the full response at once.")
    
                    TOP_P = st.number_input("Top-p", 0.0, 1.0, 0.9, 0.01, help="Top-p sampling for CHAT generation. Lower values = more focused output.")
                    TOP_K = st.number_input("Top-k", 1, 1000, 50, 1, help="Top-k sampling for CHAT generation. Lower values = more focused output.")

                    default_index = (
                        prompt_names.index("ENHANCE_QUERY")
                        if "ENHANCE_QUERY" in prompt_names else 0)

                    query_prompt_name = st.selectbox(
                        "Prompt for query enhancement",
                        options=prompt_names,
                        index=default_index,
                        key="query_prompt_selector"
                    )

                    query_prompt_text = st.session_state.prompts_dict.get(query_prompt_name,"")
                    # show the prompt
                    st.text(query_prompt_text)

# MAIN CHAT
st.title("💬 Main Query")

query = st.text_input(
    "Enter keywords and questions about the loaded documents",
    placeholder="Enter your query here",
    #help="Type your desired content here"
)

if st.session_state.enhanced_query:
    query = st.session_state.query_text
else:
    st.session_state.query_text = query

with st.container(border=False):
    if st.button("Enhance text with LLM", disabled=st.session_state.is_parsing, type="secondary", width="stretch"):
        ollama_messages =[     {
                "role": "system",
                "content": query_prompt_text
            },      {
                "role": "user",
                "content": query
            }       ]

        st.session_state.query_text = ""
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
                    st.session_state.query_text += chunk

                elif event_type == "done":
                    break

            # Render final answer
            # answer_placeholder.markdown(st.session_state.query_text)

        except requests.exceptions.Timeout:
            error_message = ("⚠️ The request to Ollama timed out.")
            st.error(error_message)

        except requests.exceptions.RequestException as e:
            error_message = (f"⚠️ Could not connect to Ollama.\n\n`{e}`")
            st.error(error_message)

        except Exception as e:
            error_message = (f"⚠️ Ollama returned an error.\n\n`{e}`")
            st.error(error_message)

        st.session_state.enhanced_query = True

    # editable answer
    if st.button("Edit / Preview enhanced query", key="Enhanced Query E/P"):
        st.session_state.query_editing = not st.session_state.query_editing
        st.rerun()

    if st.session_state.query_editing:
        st.session_state.query_text = st.text_area(
            "Markdown text",
            value=st.session_state.query_text,
            height=300,
            key="markdown_editor",
        )

    if not st.session_state.query_editing:
        st.markdown(st.session_state.query_text)

# --- EXECUTE PIPELINE ---
if st.button("🔍 Execute Advanced Search", type="primary", width="stretch", disabled=st.session_state.query_text==""):
    with st.status("Executing Retrieval Pipeline...", expanded=True) as ret_status:
        try:
            st.write(f"📡 Querying VectorDB for Top {top_k} chunks...")
            raw_results = st.session_state.vector_db.similarity_search_with_relevance_scores(st.session_state.query_text, k=top_k)
            
            st.write(f"⚖️ Applying Similarity Threshold (>= {sim_threshold})...")
            final_results = [(doc, score) for doc, score in raw_results if score >= sim_threshold]

            st.session_state.retrieval_results = final_results
            ret_status.update(label=f"Complete! Found {len(final_results)} highly relevant chunks.", state="complete", expanded=False)
            
        except Exception as e:
            st.error(f"Search failed: {e}")
            ret_status.update(label="Pipeline Failed", state="error")

    # --- RESULTS & INTERACTIVE CURATION UI ---
    if 'retrieval_results' in st.session_state and st.session_state.retrieval_results:
        st.divider()
        st.subheader("📚 Retrieved Context")
        
        grouped_docs = {}
        for doc, score in st.session_state.retrieval_results:
            src = doc.metadata.get('source', 'Unknown')
            if src not in grouped_docs:
                grouped_docs[src] = []
            grouped_docs[src].append((doc, score))

        chunk_index = 0
        
        for source_path, chunks_data in grouped_docs.items():
            filename = os.path.basename(source_path)
            dir_name = os.path.dirname(source_path)
            json_path = os.path.join(dir_name, "metadata.json")
            real_metadata = {}
            
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as jf:
                        file_meta = json.load(jf)
                        if isinstance(file_meta, dict) and filename in file_meta:
                            real_metadata = file_meta[filename]
                        elif isinstance(file_meta, list):
                            for item in file_meta:
                                if item.get('source') == filename or item.get('file') == filename:
                                    real_metadata = item
                                    break
                        else:
                            real_metadata = file_meta
                except Exception as e:
                    st.warning(f"Found metadata.json in {dir_name} but couldn't read it: {e}")
            else:
                st.warning(f"No metadata.json found in {dir_name}. Citation info may be limited.")

            st.markdown(f"### 📄 Source: `{filename}`")
            
            with st.expander(f"👁️ View Entire Document with ({len(chunks_data)} retrieved chunks highlighted)"):
                try:
                    with open(source_path, 'r', encoding='utf-8') as f:
                        full_text = f.read()
                        
                    highlighted_text = full_text
                    for c_doc, _ in chunks_data:
                        safe_chunk = re.escape(c_doc.page_content)
                        highlighted_text = re.sub(
                            f"({safe_chunk})", 
                            r"<mark style='background-color: rgba(255, 235, 59, 0.5); color: inherit; padding: 2px; border-radius: 3px;'>\1</mark>", 
                            highlighted_text
                        )
                    st.markdown(f"<div style='font-size: {st.session_state.get('global_zoom', 22)}px; height: 400px; overflow-y: auto; padding: 10px; border: 1px solid #ccc;'>{highlighted_text}</div>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Could not load or highlight original file: {e}")

            # CLEAN LOOP: Only calculating citations and calling the fragment. No duplicate UI code.
            for doc, score in chunks_data:
                doi = real_metadata.get('doi') or real_metadata.get('DOI')
                url = real_metadata.get('url') or real_metadata.get('URL')
                citation_text = ""
                
                if doi:
                    clean_doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()
                    try:
                        headers = {"Accept": "text/x-bibliography; style=apa"}
                        response = requests.get(f"https://doi.org/{clean_doi}", headers=headers, timeout=5)
                        if response.status_code == 200:
                            citation_text = response.text.strip()
                        else:
                            citation_text = f"DOI Found: {doi} (API timeout. Please use Scribbr manually)"
                    except Exception:
                        citation_text = f"DOI Found: {doi} (Fetch failed)"
                elif url:
                    citation_text = f"URL Found: {url}\n\n[🔗 Generate Citation via Scribbr](https://www.scribbr.com/citation/generator/)"
                else:
                    citation_text = f"Local File: {filename} (No DOI or URL found in metadata.json)"

                # CALL FRAGMENT (No duplicate UI elements!)
                # functions.render_chunk_ui(doc, score, real_metadata, citation_text, filename, chunk_index)
                chunk_index += 1
