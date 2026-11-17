import streamlit as st
import os
import re
import math
import time
import json
import datetime
import requests
from collections import Counter
from typing import List, Dict, Tuple, Any, Optional

# External project helpers
import functions

# LangChain & VectorDB imports
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
)
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document


# ==============================================================================
# 1. HELPER FUNCTIONS: METADATA & TEXT PROCESSING
# ==============================================================================

def parse_markdown_with_metadata(raw_text: str, source_path: str) -> Tuple[str, Dict[str, str]]:
    """
    Extracts the header block delimited by '--- METADATA ---' at the top of a file.
    Specifically parses 'DOI' and 'URL' (plus any other key: value pairs) and strips
    the metadata header from the document body so it doesn't pollute embeddings.
    """
    metadata: Dict[str, str] = {
        "source": source_path,
        "filename": os.path.basename(source_path),
        "doi": "",
        "url": "",
    }

    # Match '--- METADATA --- ... --- METADATA ---' at the start of the file
    pattern = re.compile(r"^\s*---\s*METADATA\s*---(.*?)---\s*METADATA\s*---\s*", re.DOTALL | re.IGNORECASE)
    match = pattern.search(raw_text)

    if match:
        meta_block = match.group(1)
        clean_body = raw_text[match.end():].strip()

        # Parse line-by-line key: value pairs inside the metadata block
        for line in meta_block.splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                clean_key = key.strip().lower()
                clean_val = val.strip()
                if clean_key == "doi":
                    metadata["doi"] = clean_val
                elif clean_key == "url":
                    metadata["url"] = clean_val
                elif clean_key and clean_val:
                    metadata[clean_key] = clean_val

        # Fallback regex scan inside metadata block if DOI wasn't explicitly labeled
        if not metadata["doi"]:
            doi_fallback = re.search(r"(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)", meta_block)
            if doi_fallback:
                metadata["doi"] = doi_fallback.group(1)
    else:
        clean_body = raw_text.strip()

    return clean_body, metadata


def parse_separators(raw_separators: str) -> List[str]:
    """
    Safely parses comma-separated user separator strings into actual escape characters
    while preserving intentional spaces and avoiding empty string split bugs.
    """
    parsed = []
    for item in raw_separators.split(","):
        # Unescape common whitespace literals without stripping intentional spaces
        unescaped = item.replace("\\n", "\n").replace("\\t", "\t")
        if unescaped != "":
            parsed.append(unescaped)
    return parsed if parsed else ["\n\n", "\n", ". ", " "]


def pad_chunk_to_full_sentences(chunk_text: str, full_clean_text: str) -> str:
    """
    Locates a retrieved chunk inside its parent document and expands its boundaries
    backward to the preceding sentence terminator (. ! ? or double newline) and
    forward to the following sentence terminator so no period is cut mid-sentence.
    """
    if not full_clean_text or not chunk_text:
        return chunk_text

    start_idx = full_clean_text.find(chunk_text)
    if start_idx == -1:
        # Fallback: try matching using the first 80 characters if minor whitespace differs
        anchor = chunk_text[:80]
        start_idx = full_clean_text.find(anchor)
        if start_idx == -1:
            return chunk_text

    end_idx = start_idx + len(chunk_text)

    # 1. Expand backward to find the end of the previous sentence
    search_back_limit = max(0, start_idx - 600)
    prefix_window = full_clean_text[search_back_limit:start_idx]
    back_matches = list(re.finditer(r"([.!?]\s+|\n\n)", prefix_window))
    if back_matches:
        padded_start = search_back_limit + back_matches[-1].end()
    else:
        padded_start = search_back_limit

    # 2. Expand forward to include the end of the current sentence
    search_fwd_limit = min(len(full_clean_text), end_idx + 600)
    suffix_window = full_clean_text[end_idx:search_fwd_limit]
    fwd_match = re.search(r"([.!?](?=\s|$)|\n\n)", suffix_window)
    if fwd_match:
        padded_end = end_idx + fwd_match.end()
    else:
        padded_end = search_fwd_limit

    return full_clean_text[padded_start:padded_end].strip()


# ==============================================================================
# 2. HELPER FUNCTIONS: CACHING, CITATIONS & VECTOR DB
# ==============================================================================

@st.cache_resource(show_spinner=False)
def get_vector_db(db_path: str, embedding_model: str, distance_metric: str = "Cosine Similarity") -> Chroma:
    """
    Caches the ChromaDB client and OllamaEmbeddings connection across reruns
    to prevent slow re-initialization on every UI click.
    """
    embeddings = OllamaEmbeddings(model=embedding_model)
    metric_mapping = {
        "Cosine Similarity": "cosine",
        "L2 (Euclidean)": "l2",
        "Inner Product": "ip",
    }
    collection_metadata = {"hnsw:space": metric_mapping.get(distance_metric, "cosine")}
    return Chroma(
        persist_directory=db_path,
        embedding_function=embeddings,
        collection_metadata=collection_metadata,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def resolve_citation(doi: str, url: str, filename: str) -> str:
    """
    Fetches APA bibliography formatting from doi.org with caching so repeated
    chunks from the same paper don't trigger duplicate network requests.
    """
    if doi:
        clean_doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()
        try:
            headers = {"Accept": "text/x-bibliography; style=apa"}
            resp = requests.get(f"https://doi.org/{clean_doi}", headers=headers, timeout=5)
            if resp.status_code == 200:
                return resp.text.strip()
            return f"DOI: {clean_doi} (Citation API returned status {resp.status_code})"
        except Exception:
            return f"DOI: {clean_doi} (Offline / API unreachable)"
    elif url:
        return f"URL: {url}"
    return f"Local File: {filename} (No DOI or URL in metadata header)"


# ==============================================================================
# 3. HELPER FUNCTIONS: ZERO-DEPENDENCY BM25 & HYBRID RETRIEVAL
# ==============================================================================

def _tokenize_bm25(text: str) -> List[str]:
    """Simple alphanumeric lowercase tokenizer for pure-Python BM25."""
    return re.findall(r"\b\w+\b", text.lower())


def compute_bm25_scores(query: str, corpus_texts: List[str], k1: float = 1.5, b: float = 0.75) -> List[float]:
    """
    Zero-dependency Okapi BM25 implementation over an in-memory list of document strings.
    Returns raw BM25 scores aligned by index with `corpus_texts`.
    """
    N = len(corpus_texts)
    if N == 0:
        return []

    tokenized_docs = [_tokenize_bm25(doc) for doc in corpus_texts]
    doc_lengths = [len(tokens) for tokens in tokenized_docs]
    avgdl = sum(doc_lengths) / N if N > 0 else 1.0

    # Compute Document Frequency (DF) for each term
    df: Counter = Counter()
    tf_list: List[Counter] = []
    for tokens in tokenized_docs:
        counts = Counter(tokens)
        tf_list.append(counts)
        for term in counts.keys():
            df[term] += 1

    query_tokens = _tokenize_bm25(query)
    scores = [0.0] * N

    for q_term in query_tokens:
        if q_term not in df:
            continue
        # Standard Robertson-Spärck Jones IDF formula
        n_q = df[q_term]
        idf = math.log((N - n_q + 0.5) / (n_q + 0.5) + 1.0)

        for idx in range(N):
            freq = tf_list[idx].get(q_term, 0)
            if freq == 0:
                continue
            dl = doc_lengths[idx]
            numerator = freq * (k1 + 1.0)
            denominator = freq + k1 * (1.0 - b + b * (dl / avgdl))
            scores[idx] += idf * (numerator / denominator)

    return scores


def hybrid_search(
    vector_db: Chroma,
    search_queries: List[str],
    top_k: int,
    sim_threshold: float,
    alpha: float,
) -> List[Tuple[Document, float]]:
    """
    Combines Vector Semantic Search and Pure-Python BM25 Keyword Search using
    convex combination: `final_score = alpha * semantic_score + (1 - alpha) * bm25_norm`.
    Supports multiple query variants (e.g., original + Step-Back + HyDE).
    """
    # 1. Fetch all stored chunks from Chroma for BM25 scoring & candidate pooling
    db_dump = vector_db.get(include=["documents", "metadatas"])
    all_texts: List[str] = db_dump.get("documents") or []
    all_metas: List[dict] = db_dump.get("metadatas") or []

    if not all_texts:
        return []

    # Build a lookup map from chunk text -> Document + best semantic score across query variants
    semantic_scores: Dict[str, float] = {}
    doc_lookup: Dict[str, Document] = {}

    for txt, meta in zip(all_texts, all_metas):
        doc_lookup[txt] = Document(page_content=txt, metadata=meta or {})
        semantic_scores[txt] = 0.0

    # Run semantic similarity search if alpha > 0
    if alpha > 0.0:
        candidate_k = min(len(all_texts), max(top_k * 3, 25))
        for q in search_queries:
            try:
                results = vector_db.similarity_search_with_relevance_scores(q, k=candidate_k)
                for doc, score in results:
                    # Clamp relevance score safely to [0, 1] across distance metrics
                    clamped_score = max(0.0, min(1.0, float(score)))
                    if clamped_score > semantic_scores.get(doc.page_content, 0.0):
                        semantic_scores[doc.page_content] = clamped_score
            except Exception:
                # Fallback if relevance score normalization throws on custom distance spaces
                raw_docs = vector_db.similarity_search(q, k=candidate_k)
                for rank, doc in enumerate(raw_docs):
                    fallback_score = 1.0 - (rank / max(1, candidate_k))
                    if fallback_score > semantic_scores.get(doc.page_content, 0.0):
                        semantic_scores[doc.page_content] = fallback_score

    # 2. Run BM25 keyword scoring across all chunks if alpha < 1.0
    bm25_scores_map: Dict[str, float] = {txt: 0.0 for txt in all_texts}
    if alpha < 1.0:
        combined_query = " ".join(search_queries)
        raw_bm25 = compute_bm25_scores(combined_query, all_texts)
        max_bm25 = max(raw_bm25) if raw_bm25 else 0.0
        if max_bm25 > 0:
            for txt, raw_val in zip(all_texts, raw_bm25):
                bm25_scores_map[txt] = raw_val / max_bm25  # Min-Max normalize to [0, 1]

    # 3. Fuse scores using alpha weight & filter by threshold
    fused_results: List[Tuple[Document, float]] = []
    for txt, doc in doc_lookup.items():
        sem_s = semantic_scores.get(txt, 0.0)
        kw_s = bm25_scores_map.get(txt, 0.0)
        hybrid_score = (alpha * sem_s) + ((1.0 - alpha) * kw_s)

        if hybrid_score >= sim_threshold:
            fused_results.append((doc, round(hybrid_score, 4)))

    # Sort descending by hybrid score and slice top_k
    fused_results.sort(key=lambda x: x[1], reverse=True)
    return fused_results[:top_k]


# ==============================================================================
# 4. HELPER FUNCTIONS: LOCAL OLLAMA LLM OPERATIONS
# ==============================================================================

def call_ollama_sync(
    messages: List[Dict[str, str]],
    ollama_url: str,
    model: str,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    timeout: int = 180,
) -> str:
    """
    Synchronous helper wrapping `functions.stream_ollama_chat` to collect the full
    response text cleanly for background pipeline steps (HyDE, Step-Back, OCR clean, Rerank).
    """
    output_text = ""
    for event in functions.stream_ollama_chat(
        messages=messages,
        ollama_url=ollama_url,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        top_p=0.9,
        top_k=40,
        thinking=False,
    ):
        if event.get("type") == "content":
            output_text += event.get("content", "")
        elif event.get("type") == "done":
            break
    return output_text.strip()


def generate_query_expansions(
    base_query: str,
    use_hyde: bool,
    use_step_back: bool,
    ollama_url: str,
    model: str,
    timeout: int,
) -> Tuple[List[str], Dict[str, str]]:
    """
    Generates Step-Back and/or HyDE search expansions using the local Ollama model.
    Returns the list of queries to search and a dict of generated artifacts for UI display.
    """
    queries = [base_query]
    artifacts: Dict[str, str] = {}

    if use_step_back:
        sb_messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert research assistant. Given a specific user query, "
                    "generate a broader, high-level 'step-back' question that captures the "
                    "underlying conceptual principles or background context needed to answer it. "
                    "Return ONLY the step-back question, nothing else."
                ),
            },
            {"role": "user", "content": base_query},
        ]
        sb_query = call_ollama_sync(sb_messages, ollama_url, model, temperature=0.3, timeout=timeout)
        if sb_query:
            queries.append(sb_query)
            artifacts["Step-Back Query"] = sb_query

    if use_hyde:
        hyde_messages = [
            {
                "role": "system",
                "content": (
                    "You are an academic domain expert. Write a concise, factual 1-paragraph "
                    "hypothetical excerpt from a research paper or technical manual that directly "
                    "answers the user's query. Do not include conversational filler—write only the excerpt."
                ),
            },
            {"role": "user", "content": base_query},
        ]
        hyde_doc = call_ollama_sync(hyde_messages, ollama_url, model, temperature=0.4, timeout=timeout)
        if hyde_doc:
            queries.append(hyde_doc)
            artifacts["HyDE Hypothetical Passage"] = hyde_doc

    return queries, artifacts


def clean_ocr_with_llm(text: str, ollama_url: str, model: str, timeout: int) -> str:
    """
    Uses the local Ollama model to fix OCR artifacts (broken line hyphenation,
    stray symbols, merged/garbled characters) while strictly preserving facts and wording.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a meticulous copy-editor fixing OCR extraction artifacts in academic/technical text. "
                "Fix broken hyphenated line-breaks, stray OCR symbols, and spacing errors. "
                "STRICT RULES: Do NOT summarize, do NOT add new information, do NOT alter technical terms, "
                "numbers, or citations. Output ONLY the cleaned text."
            ),
        },
        {"role": "user", "content": text},
    ]
    cleaned = call_ollama_sync(messages, ollama_url, model, temperature=0.1, timeout=timeout)
    return cleaned if cleaned else text


def llm_select_and_clean_top_n(
    query: str,
    chunks: List[Dict[str, Any]],
    top_n: int,
    ollama_url: str,
    model: str,
    timeout: int,
) -> List[Dict[str, Any]]:
    """
    Asks the local Ollama LLM to evaluate all retrieved chunks against the query,
    select the best `top_n` chunks, and clean OCR artifacts on the selected winners.
    """
    numbered_candidates = "\n\n".join(
        [f"[CHUNK {i}] (Source: {c['filename']}, Score: {c['score']}):\n{c['content']}" for i, c in enumerate(chunks)]
    )

    selection_messages = [
        {
            "role": "system",
            "content": (
                f"You are a precision RAG reranker. Given a user query and numbered text chunks, "
                f"select the {top_n} most relevant, non-redundant chunks that best answer the query. "
                "Respond ONLY with valid JSON in this exact format:\n"
                '{"selected": [{"index": 0, "reason": "1-sentence justification"}]}'
            ),
        },
        {
            "role": "user",
            "content": f"QUERY: {query}\n\nCANDIDATE CHUNKS:\n{numbered_candidates}",
        },
    ]

    raw_response = call_ollama_sync(selection_messages, ollama_url, model, temperature=0.1, timeout=timeout)

    # Extract JSON block safely even if the LLM wraps it in markdown fences
    selected_map: Dict[int, str] = {}
    json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            for item in parsed.get("selected", [])[:top_n]:
                idx = int(item["index"])
                selected_map[idx] = item.get("reason", "Selected by LLM reranker.")
        except Exception:
            pass

    # Fallback if JSON parsing failed: pick top N by retrieval score
    if not selected_map:
        for i in range(min(top_n, len(chunks))):
            selected_map[i] = "Selected via top retrieval score (LLM JSON fallback)."

    # Update selection state and run OCR cleanup on the chosen chunks
    for i, chunk in enumerate(chunks):
        if i in selected_map:
            chunk["selected"] = True
            chunk["llm_reason"] = selected_map[i]
            chunk["content"] = clean_ocr_with_llm(chunk["content"], ollama_url, model, timeout)
        else:
            chunk["selected"] = False
            chunk["llm_reason"] = ""

    return chunks


# ==============================================================================
# 5. HELPER FUNCTIONS: OBSIDIAN EXPORT
# ==============================================================================

def build_obsidian_markdown(
    query: str,
    selected_chunks: List[Dict[str, Any]],
    generated_answer: str = "",
) -> str:
    """
    Formats curated chunks (and optional synthesized answer) into an Obsidian-friendly
    Markdown document with YAML frontmatter, callouts, citations, and [[wikilinks]].
    """
    today_str = datetime.date.today().isoformat()
    lines = [
        "---",
        f"created: {today_str}",
        f'query: "{query.replace(chr(34), chr(39))}"',
        "tags: [rag-research, curated-sources]",
        "---",
        "",
        f"# Research Synthesis: {query}",
        "",
    ]

    if generated_answer.strip():
        lines.extend([
            "## 🧠 Synthesized Answer",
            "",
            generated_answer.strip(),
            "",
            "---",
            "",
        ])

    lines.append("## 📚 Curated Source Excerpts\n")

    # Group selected chunks by source file for clean Obsidian backlinking
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for ch in selected_chunks:
        grouped.setdefault(ch["filename"], []).append(ch)

    for filename, items in grouped.items():
        # Strip .md extension for clean Obsidian [[wikilink]]
        note_title = os.path.splitext(filename)[0]
        first = items[0]

        lines.append(f"### [[{note_title}]]")
        if first.get("doi"):
            lines.append(f"- **DOI:** `{first['doi']}`")
        if first.get("url"):
            lines.append(f"- **URL:** {first['url']}")
        if first.get("citation"):
            lines.append(f"- **Citation:** {first['citation']}")
        lines.append("")

        for idx, item in enumerate(items, start=1):
            reason_note = f" | *LLM Note: {item['llm_reason']}*" if item.get("llm_reason") else ""
            lines.append(f"> [!quote] Excerpt {idx} (Relevance: {item['score']:.2f}){reason_note}")
            for paragraph_line in item["content"].splitlines():
                lines.append(f"> {paragraph_line}")
            lines.append("")

    return "\n".join(lines)


# ==============================================================================
# 6. INITIALIZATION & SESSION STATE
# ==============================================================================

prompt_names = list(st.session_state.prompts_dict.keys())
ollama_models = st.session_state.ollama_models
OLLAMA_HOST = st.session_state.ollama_host
MD_FOLDER = st.session_state.folder_paths["MD_FOLDER"]
DB_PATH = st.session_state.folder_paths["DB_FOLDER"]

os.makedirs(DB_PATH, exist_ok=True)
os.makedirs(MD_FOLDER, exist_ok=True)

# Initialize session state keys safely
defaults = {
    "rag_config": {},
    "query_text": "",
    "enhanced_query": False,
    "query_editing": False,
    "vector_db": None,
    "curated_chunks": [],
    "query_expansions": {},
    "final_rag_answer": "",
    "final_rag_thinking": "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ==============================================================================
# 7. SIDEBAR: DATABASE INGESTION & RAG PARAMETERS
# ==============================================================================

with st.sidebar:
    with st.expander("📚 RAG Settings & VectorDB", expanded=True):
        embedding_idx = (
            ollama_models.index("nomic-embed-text-v2-moe:latest")
            if "nomic-embed-text-v2-moe:latest" in ollama_models
            else 0
        )
        embedding_model = st.selectbox("Embedding Model", ollama_models, index=embedding_idx)
        st.session_state.rag_config["embedding_model"] = embedding_model

        # --- EMBEDDING & CHUNKING PARAMETERS ---
        with st.expander("Embedding & Chunking Parameters", expanded=False):
            chunk_strategy = st.selectbox(
                "Chunking Strategy", ["Recursive Character", "Token-Based", "Strict Character"]
            )
            chunk_size = st.number_input("Chunk Size", min_value=100, max_value=8000, value=1300, step=100)
            chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=2000, value=350, step=50)

            if chunk_strategy in ["Recursive Character", "Strict Character"]:
                raw_separators = st.text_input(
                    "Separators (comma-separated, highest to lowest priority):",
                    value="\\n\\n, \\n, . ,  , #, ##, ###",
                )
                separators = parse_separators(raw_separators)
            else:
                st.info("Separators are handled automatically by the tokenizer.")
                separators = None

            distance_metric = st.selectbox(
                "Distance Metric", ["Cosine Similarity", "L2 (Euclidean)", "Inner Product"]
            )

            st.session_state.rag_config.update({
                "chunk_strategy": chunk_strategy,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "separators": separators,
                "distance_metric": distance_metric,
            })

        # --- AUTO-LOAD EXISTING VECTOR DB (FIXED CONDITION) ---
        if st.session_state.vector_db is None and os.path.exists(DB_PATH) and len(os.listdir(DB_PATH)) > 0:
            try:
                st.session_state.vector_db = get_vector_db(DB_PATH, embedding_model, distance_metric)
            except Exception as e:
                st.error(f"Could not auto-load existing DB: {e}")

        if st.session_state.vector_db is not None:
            st.success(f"🟢 Active DB: `{os.path.basename(DB_PATH)}`")
        else:
            st.warning("🔴 No active database loaded yet.")

        # --- INGESTION & DATABASE CREATION BUTTON ---
        if st.button(
            "Save Config & Build / Update DB",
            help="Parses '--- METADATA ---' headers, embeds new .md files, and appends them to ChromaDB.",
            use_container_width=True,
        ):
            with st.spinner("Scanning & indexing Markdown documents..."):
                try:
                    vector_db = get_vector_db(DB_PATH, embedding_model, distance_metric)

                    loader = DirectoryLoader(
                        MD_FOLDER,
                        glob="**/*.md",
                        exclude=["**/chroma_db/**"],
                        loader_cls=TextLoader,
                        loader_kwargs={"encoding": "utf-8"},
                    )
                    raw_docs = loader.load()

                    if not raw_docs:
                        st.warning(f"No .md files found in `{MD_FOLDER}`.")
                        st.stop()

                    # Filter out files already indexed in Chroma
                    existing_sources = set()
                    try:
                        existing_data = vector_db.get(include=["metadatas"])
                        if existing_data and existing_data.get("metadatas"):
                            existing_sources = {
                                m.get("source") for m in existing_data["metadatas"] if m and m.get("source")
                            }
                    except Exception as e:
                        st.warning(f"Metadata check warning: {e}")

                    new_raw_docs = [d for d in raw_docs if d.metadata.get("source") not in existing_sources]

                    if not new_raw_docs:
                        st.success("✅ Database is already up to date!")
                        st.session_state.vector_db = vector_db
                        time.sleep(0.8)
                        st.rerun()

                    # Configure text splitter
                    if chunk_strategy == "Recursive Character":
                        splitter = RecursiveCharacterTextSplitter(
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            separators=separators,
                        )
                    elif chunk_strategy == "Strict Character":
                        splitter = CharacterTextSplitter(
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            separator=separators[0] if separators else "\n\n",
                        )
                    else:
                        splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

                    progress_bar = st.progress(0)
                    for i, raw_doc in enumerate(new_raw_docs):
                        src_path = raw_doc.metadata.get("source", "")
                        # Parse '--- METADATA ---' block and strip it before chunking
                        clean_body, parsed_meta = parse_markdown_with_metadata(raw_doc.page_content, src_path)

                        clean_doc = Document(page_content=clean_body, metadata=parsed_meta)
                        chunks = splitter.split_documents([clean_doc])
                        if chunks:
                            vector_db.add_documents(chunks)

                        progress_bar.progress((i + 1) / len(new_raw_docs))

                    st.session_state.vector_db = vector_db
                    st.success(f"✅ Indexed {len(new_raw_docs)} new file(s)!")
                    time.sleep(0.8)
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Indexing error: {e}")

        # --- RETRIEVAL & LLM PARAMETERS ---
        with st.expander("RAG & LLM Parameters", expanded=False):
            top_k = st.number_input("Initial Retrieval Top-K", min_value=1, max_value=50, value=10)
            sim_threshold = st.slider("Similarity Threshold", min_value=0.0, max_value=1.0, value=0.35, step=0.05)
            alpha_weight = st.slider(
                "Hybrid Alpha (α)",
                min_value=0.0,
                max_value=1.0,
                value=0.6,
                step=0.05,
                help="0.0 = Pure BM25 Keyword | 1.0 = Pure Vector Semantic",
            )

            use_hyde = st.toggle("Enable HyDE", value=False, help="Generates a hypothetical answer passage for semantic matching.")
            use_step_back = st.toggle("Enable Step-Back", value=False, help="Generates a broader conceptual query alongside your query.")

            with st.expander("Model Settings", expanded=False, icon="⚙️"):
                text_idx = ollama_models.index("granite4.2:3b") if "granite4.2:3b" in ollama_models else 0
                MODEL = st.selectbox("Main Model", ollama_models, index=text_idx, key="RAG_model")
                OLLAMA_URL = st.text_input("Ollama URL", value=f"{OLLAMA_HOST}/api/chat")
                TEMPERATURE = st.number_input("Temperature", 0.0, 3.0, 0.3, 0.1)
                MAX_TOKENS = st.number_input("Max Tokens", -1, 100000, 10000, 1)
                REQUEST_TIMEOUT = st.number_input("Request Timeout (s)", 1, 1000, 300, 1)
                STREAMING = st.toggle("Streaming Generation", value=True, key="chat_streaming")
                TOP_P = st.number_input("Top-p", 0.0, 1.0, 0.9, 0.01)
                TOP_K_GEN = st.number_input("Top-k", 1, 1000, 50, 1)

                default_idx = prompt_names.index("ENHANCE_QUERY") if "ENHANCE_QUERY" in prompt_names else 0
                query_prompt_name = st.selectbox("Query Enhancement Prompt", options=prompt_names, index=default_idx)
                query_prompt_text = st.session_state.prompts_dict.get(query_prompt_name, "")
                st.caption(query_prompt_text)


# ==============================================================================
# 8. MAIN INTERFACE: QUERY INPUT & ENHANCEMENT
# ==============================================================================

st.title("💬 Research RAG Workbench")

raw_input_query = st.text_input(
    "Enter keywords or research questions about your documents:",
    placeholder="e.g., How does mitochondrial dysfunction affect synaptic plasticity?",
)

# Sync raw input unless the user has explicitly generated/edited an enhanced query
if not st.session_state.enhanced_query:
    st.session_state.query_text = raw_input_query

col_enh, col_edit = st.columns([1, 1])
with col_enh:
    if st.button("✨ Enhance Query with LLM", disabled=st.session_state.get("is_parsing", False), use_container_width=True):
        if not raw_input_query.strip():
            st.warning("Enter a base query first.")
        else:
            ollama_messages = [
                {"role": "system", "content": query_prompt_text},
                {"role": "user", "content": raw_input_query},
            ]
            st.session_state.query_text = ""
            st.session_state.thinking_text = ""

            thinking_expander = st.expander("🧠 Query Enhancement Reasoning", expanded=False)
            with thinking_expander:
                thinking_placeholder = st.empty()

            try:
                for event in functions.stream_ollama_chat(
                    messages=ollama_messages,
                    ollama_url=OLLAMA_URL,
                    model=MODEL,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    timeout=REQUEST_TIMEOUT,
                    top_p=TOP_P,
                    top_k=TOP_K_GEN,
                    thinking=True,
                ):
                    etype = event["type"]
                    chunk = event.get("content", "")
                    if etype == "thinking":
                        st.session_state.thinking_text += chunk
                        thinking_placeholder.markdown(st.session_state.thinking_text)
                    elif etype == "content":
                        st.session_state.query_text += chunk
                    elif etype == "done":
                        break
                st.session_state.enhanced_query = True
            except Exception as e:
                st.error(f"⚠️ Ollama error during query enhancement: {e}")

with col_edit:
    if st.button("✏️ Edit / Preview Query", use_container_width=True):
        st.session_state.query_editing = not st.session_state.query_editing
        st.rerun()

if st.session_state.query_editing:
    st.session_state.query_text = st.text_area(
        "Active Search Query (Markdown):",
        value=st.session_state.query_text,
        height=150,
    )
elif st.session_state.enhanced_query and st.session_state.query_text:
    st.info(f"**Active Enhanced Query:** {st.session_state.query_text}")


# ==============================================================================
# 9. EXECUTION: HYBRID RETRIEVAL + PERIOD PADDING
# ==============================================================================

if st.button(
    "🔍 Execute Advanced Search",
    type="primary",
    use_container_width=True,
    disabled=(not st.session_state.query_text.strip() or st.session_state.vector_db is None),
):
    with st.status("Executing Advanced Retrieval Pipeline...", expanded=True) as ret_status:
        try:
            active_query = st.session_state.query_text.strip()

            # Step 1: Query Transformations (Step-Back & HyDE)
            if use_hyde or use_step_back:
                st.write("🧠 Generating Step-Back / HyDE query expansions...")
            search_queries, expansions = generate_query_expansions(
                base_query=active_query,
                use_hyde=use_hyde,
                use_step_back=use_step_back,
                ollama_url=OLLAMA_URL,
                model=MODEL,
                timeout=REQUEST_TIMEOUT,
            )
            st.session_state.query_expansions = expansions

            # Step 2: Hybrid Search (Vector + Pure-Python BM25)
            st.write(f"📡 Running Hybrid Search (α={alpha_weight:.2f}, Top-K={top_k}, Threshold≥{sim_threshold:.2f})...")
            final_results = hybrid_search(
                vector_db=st.session_state.vector_db,
                search_queries=search_queries,
                top_k=top_k,
                sim_threshold=sim_threshold,
                alpha=alpha_weight,
            )

            # Step 3: Load parent files once to pad chunks to complete sentences/periods
            st.write("📐 Padding chunks to complete sentence boundaries & resolving citations...")
            clean_docs_cache: Dict[str, str] = {}
            curated_list: List[Dict[str, Any]] = []

            for idx, (doc, score) in enumerate(final_results):
                src_path = doc.metadata.get("source", "Unknown")
                filename = doc.metadata.get("filename") or os.path.basename(src_path)
                doi = doc.metadata.get("doi", "")
                url = doc.metadata.get("url", "")

                # Load & strip metadata from source file once per unique path
                if src_path not in clean_docs_cache:
                    if os.path.exists(src_path):
                        with open(src_path, "r", encoding="utf-8") as f:
                            body, file_meta = parse_markdown_with_metadata(f.read(), src_path)
                            clean_docs_cache[src_path] = body
                            # Backfill DOI/URL if DB was built with an older version
                            doi = doi or file_meta.get("doi", "")
                            url = url or file_meta.get("url", "")
                    else:
                        clean_docs_cache[src_path] = ""

                padded_content = pad_chunk_to_full_sentences(doc.page_content, clean_docs_cache[src_path])
                citation_str = resolve_citation(doi, url, filename)

                curated_list.append({
                    "id": f"chunk_{idx}",
                    "selected": True,
                    "content": padded_content,
                    "raw_content": doc.page_content,
                    "score": score,
                    "source": src_path,
                    "filename": filename,
                    "doi": doi,
                    "url": url,
                    "citation": citation_str,
                    "llm_reason": "",
                })

            st.session_state.curated_chunks = curated_list
            st.session_state.final_rag_answer = ""  # Reset previous answer on new search
            ret_status.update(
                label=f"✅ Retrieval Complete: {len(curated_list)} period-padded chunks ready.",
                state="complete",
                expanded=False,
            )
        except Exception as e:
            st.error(f"Search pipeline failed: {e}")
            ret_status.update(label="Pipeline Failed", state="error")


# ==============================================================================
# 10. PERSISTENT CURATION UI, LLM AUTO-SELECTOR, EXPORT & FINAL GENERATION
# ==============================================================================
# NOTE: Kept OUTSIDE the search button block so edits & selections survive reruns!

if st.session_state.curated_chunks:
    st.divider()

    # Show HyDE / Step-Back expansions if generated
    if st.session_state.query_expansions:
        with st.expander("🔎 View Query Expansions (HyDE / Step-Back)", expanded=False):
            for exp_title, exp_text in st.session_state.query_expansions.items():
                st.markdown(f"**{exp_title}:** {exp_text}")

    st.subheader(f"📚 Retrieved & Period-Padded Context ({len(st.session_state.curated_chunks)} chunks)")

    # --- AUTOMATED LLM CURATION BAR ---
    with st.container(border=True):
        c1, c2, c3 = st.columns([1.2, 2.3, 1.5])
        with c1:
            best_n = st.number_input(
                "Best N chunks",
                min_value=1,
                max_value=len(st.session_state.curated_chunks),
                value=min(3, len(st.session_state.curated_chunks)),
            )
        with c2:
            st.write("")
            st.write("")
            if st.button("🤖 Auto-Select Best N & Clean OCR", use_container_width=True):
                with st.spinner(f"Ollama is selecting the top {best_n} chunks and cleaning OCR artifacts..."):
                    st.session_state.curated_chunks = llm_select_and_clean_top_n(
                        query=st.session_state.query_text,
                        chunks=st.session_state.curated_chunks,
                        top_n=best_n,
                        ollama_url=OLLAMA_URL,
                        model=MODEL,
                        timeout=REQUEST_TIMEOUT,
                    )
                    # Sync widget states so checkboxes and text_areas reflect the updated values
                    for ch in st.session_state.curated_chunks:
                        st.session_state[f"chk_{ch['id']}"] = ch["selected"]
                        st.session_state[f"txt_{ch['id']}"] = ch["content"]
                    st.rerun()
        with c3:
            st.write("")
            st.write("")
            if st.button("☑️ Select All", use_container_width=True):
                for ch in st.session_state.curated_chunks:
                    ch["selected"] = True
                    st.session_state[f"chk_{ch['id']}"] = True
                st.rerun()

    # --- GROUPED SOURCE & CHUNK EDITOR ---
    grouped_indices: Dict[str, List[int]] = {}
    for idx, ch in enumerate(st.session_state.curated_chunks):
        grouped_indices.setdefault(ch["source"], []).append(idx)

    for src_path, idx_list in grouped_indices.items():
        first_chunk = st.session_state.curated_chunks[idx_list[0]]
        filename = first_chunk["filename"]

        st.markdown(f"### 📄 Source: `{filename}`")
        st.caption(f"📖 {first_chunk['citation']}")

        # Full document viewer with highlighted retrieved passages
        with st.expander(f"👁️ View Full Clean Document ({len(idx_list)} matched passage(s))", expanded=False):
            if os.path.exists(src_path):
                with open(src_path, "r", encoding="utf-8") as f:
                    clean_full_doc, _ = parse_markdown_with_metadata(f.read(), src_path)
                highlighted_doc = clean_full_doc
                for i in idx_list:
                    target_txt = st.session_state.curated_chunks[i]["raw_content"]
                    safe_pattern = re.escape(target_txt)
                    highlighted_doc = re.sub(
                        f"({safe_pattern})",
                        r"<mark style='background-color: rgba(255, 235, 59, 0.55); padding: 2px; border-radius: 3px;'>\1</mark>",
                        highlighted_doc,
                    )
                zoom_px = st.session_state.get("global_zoom", 18)
                st.markdown(
                    f"<div style='font-size:{zoom_px}px; height:350px; overflow-y:auto; padding:12px; border:1px solid #444; border-radius:6px;'>{highlighted_doc}</div>",
                    unsafe_allow_html=True,
                )

        # Individual chunk cards (Checkbox + Editable Text + Single-Chunk OCR Clean)
        for i in idx_list:
            ch = st.session_state.curated_chunks[i]
            with st.container(border=True):
                top_row_l, top_row_r = st.columns([3, 1])
                with top_row_l:
                    is_sel = st.checkbox(
                        f"Include Chunk #{i + 1} (Relevance Score: {ch['score']:.3f})",
                        value=ch["selected"],
                        key=f"chk_{ch['id']}",
                    )
                    st.session_state.curated_chunks[i]["selected"] = is_sel
                with top_row_r:
                    if st.button("✨ Clean OCR", key=f"ocr_{ch['id']}", use_container_width=True):
                        with st.spinner("Cleaning OCR..."):
                            cleaned = clean_ocr_with_llm(ch["content"], OLLAMA_URL, MODEL, REQUEST_TIMEOUT)
                            st.session_state.curated_chunks[i]["content"] = cleaned
                            st.session_state[f"txt_{ch['id']}"] = cleaned
                            st.rerun()

                if ch.get("llm_reason"):
                    st.info(f"🤖 **LLM Selection Note:** {ch['llm_reason']}")

                edited_text = st.text_area(
                    "Chunk Text (Editable):",
                    value=ch["content"],
                    height=160,
                    key=f"txt_{ch['id']}",
                    label_visibility="collapsed",
                )
                st.session_state.curated_chunks[i]["content"] = edited_text

    # ==========================================================================
    # 11. FINAL ANSWER GENERATION & OBSIDIAN MARKDOWN EXPORT
    # ==========================================================================
    st.divider()
    selected_chunks = [c for c in st.session_state.curated_chunks if c["selected"]]
    st.markdown(f"**Selected for Synthesis & Export:** `{len(selected_chunks)}` of `{len(st.session_state.curated_chunks)}` chunks")

    # --- FINAL GENERATION BUTTON ---
    if st.button(
        "🧠 Generate Final Answer from Selected Chunks",
        type="primary",
        use_container_width=True,
        disabled=(len(selected_chunks) == 0),
    ):
        context_blocks = []
        for idx, c in enumerate(selected_chunks, start=1):
            note_link = os.path.splitext(c["filename"])[0]
            context_blocks.append(
                f"[Source {idx}: [[{note_link}]] | {c['citation']}]\n{c['content']}"
            )
        compiled_context = "\n\n---\n\n".join(context_blocks)

        rag_messages = [
            {
                "role": "system",
                "content": (
                    "You are a rigorous research assistant. Answer the user's query using ONLY "
                    "the provided curated context excerpts. Cite sources inline using Obsidian wikilinks "
                    "like [[Filename]] whenever you state a claim. If the context does not contain enough "
                    "information to answer part of the question, explicitly state what is missing."
                ),
            },
            {
                "role": "user",
                "content": f"CURATED CONTEXT:\n{compiled_context}\n\nUSER QUERY:\n{st.session_state.query_text}",
            },
        ]

        st.session_state.final_rag_answer = ""
        st.session_state.final_rag_thinking = ""

        ans_thinking_exp = st.expander("🧠 Final Generation Reasoning", expanded=False)
        with ans_thinking_exp:
            ans_thinking_ph = st.empty()
        ans_output_ph = st.empty()

        try:
            for event in functions.stream_ollama_chat(
                messages=rag_messages,
                ollama_url=OLLAMA_URL,
                model=MODEL,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                timeout=REQUEST_TIMEOUT,
                top_p=TOP_P,
                top_k=TOP_K_GEN,
                thinking=True,
            ):
                etype = event["type"]
                chunk = event.get("content", "")
                if etype == "thinking":
                    st.session_state.final_rag_thinking += chunk
                    ans_thinking_ph.markdown(st.session_state.final_rag_thinking)
                elif etype == "content":
                    st.session_state.final_rag_answer += chunk
                    ans_output_ph.markdown(st.session_state.final_rag_answer)
                elif etype == "done":
                    break
        except Exception as e:
            st.error(f"⚠️ Final generation failed: {e}")

    # Persist generated answer across reruns (e.g., when user clicks Export right after)
    if st.session_state.final_rag_answer:
        with st.container(border=True):
            st.markdown("### 📝 Synthesized Research Answer")
            st.markdown(st.session_state.final_rag_answer)

    # --- OBSIDIAN MARKDOWN EXPORT ---
    with st.expander("💾 Save Curated Chunks to Obsidian Markdown (.md)", expanded=True):
        default_slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", st.session_state.query_text[:40]).strip("_") or "rag_notes"
        export_filename = st.text_input(
            "Output Filename (.md):",
            value=f"{datetime.date.today().isoformat()}_{default_slug}.md",
        )
        include_answer_in_export = st.checkbox(
            "Include Synthesized Answer at top of exported note",
            value=bool(st.session_state.final_rag_answer),
        )

        obsidian_md_content = build_obsidian_markdown(
            query=st.session_state.query_text,
            selected_chunks=selected_chunks,
            generated_answer=st.session_state.final_rag_answer if include_answer_in_export else "",
        )

        exp_col1, exp_col2 = st.columns([1, 1])
        with exp_col1:
            if st.button(
                f"💾 Save to `{os.path.basename(MD_FOLDER)}`",
                use_container_width=True,
                disabled=(len(selected_chunks) == 0),
            ):
                safe_name = export_filename if export_filename.endswith(".md") else f"{export_filename}.md"
                save_path = os.path.join(MD_FOLDER, safe_name)
                try:
                    with open(save_path, "w", encoding="utf-8") as out_f:
                        out_f.write(obsidian_md_content)
                    st.success(f"✅ Saved Obsidian note to `{save_path}`!")
                except Exception as e:
                    st.error(f"❌ Could not save file: {e}")

        with exp_col2:
            st.download_button(
                label="⬇️ Download .md Copy",
                data=obsidian_md_content,
                file_name=export_filename if export_filename.endswith(".md") else f"{export_filename}.md",
                mime="text/markdown",
                use_container_width=True,
                disabled=(len(selected_chunks) == 0),
            )
            