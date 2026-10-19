import os
import base64
import requests
from pdf2image import convert_from_path
import json
import streamlit as st
import subprocess
import signal
import time
from typing import List, Tuple
from pathlib import Path

import psutil
## da fare il grafico delle performance dell'hardware

# ---------------------------------------------------------
# system functions management
# ---------------------------------------------------------

def add_log(message: str):
    """Appends a timestamped log to the session state."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.app_logs.insert(0, f"[{timestamp}] {message}")

def list_of_filetype(source_dir: Path, filetype: str) -> list[Path]:
    """Find PDF files, optionally limiting how deep the scan goes."""
    if not source_dir.is_dir():
        return []

    candidates = source_dir.rglob("*")
    fileoftype: list[Path] = []

    for path in candidates:
        if path.is_file() and f"{filetype}".lower() in path.suffix.lower():
            fileoftype.append(path)

    return sorted(fileoftype, key=lambda path: str(path.relative_to(source_dir)).lower())

def load_json_as_booldict(selection_file: Path) -> dict[str, bool]:
    """Load saved per-file selections. Return an empty mapping if unavailable."""
    try:
        if selection_file.exists():
            data = json.loads(selection_file.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {str(key): bool(value) for key, value in data.items()}
    except (OSError, json.JSONDecodeError):
        pass

    return {}

def save_booldict_as_json(selection_filepath: Path, selection: dict[str, bool]) -> None:
    """Persist selected files as JSON."""
    try:
        selection_filepath.parent.mkdir(parents=True, exist_ok=True)
        selection_filepath.write_text(
            json.dumps(selection, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except OSError as exc:
        st.warning(f"Could not save the selection: {exc}")

# ---------------------------------------------------------
# Ollama process management
# ---------------------------------------------------------

def ollama_is_running(OLLAMA_HOST: str, verbose: bool) -> bool:
    """Check whether an Ollama server is responding."""
    try:
        response = requests.get(
            f"{OLLAMA_HOST}/api/tags",
            timeout=1,
        )
        return response.status_code == 200

    except requests.RequestException as e:
        if verbose:
            st.warning(f"Error checking Ollama status on {OLLAMA_HOST}")
            st.warning(f"Error: {str(e)}")
        return False


def start_ollama(OLLAMA_HOST: str, OLLAMA_CONTEXT_LENGTH: int):
    """
    Start Ollama with the requested context length.

    Equivalent to:
        OLLAMA_CONTEXT_LENGTH=N ollama serve
    """
    if ollama_is_running(OLLAMA_HOST=OLLAMA_HOST, verbose=False):
        print("Ollama is already running.")
        return None

    env = os.environ.copy()

    # This is equivalent to:
    # OLLAMA_CONTEXT_LENGTH=1024 ollama serve
    env["OLLAMA_CONTEXT_LENGTH"] = str(OLLAMA_CONTEXT_LENGTH)

    process = subprocess.Popen(
        ["ollama", "serve"],
        env=env,
        # Prevent Ollama output from filling Streamlit's process output.
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,

        # Give us a separate process group so it can be reliably terminated later.
        start_new_session=True,
    )

    # Wait for Ollama to become available.
    for _ in range(50):
        if ollama_is_running(OLLAMA_HOST, verbose=False):
            print(f"Ollama is now running with context lenght {OLLAMA_CONTEXT_LENGTH}")
            return process

        # Detect immediate startup failure.
        if process.poll() is not None:
            raise RuntimeError(
                f"Ollama exited immediately with code "
                f"{process.returncode}"
            )

        time.sleep(0.2)

    # Timed out.
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass

    raise RuntimeError(
        "Ollama did not become available within the expected time, process killed. Check that Ollama is installed and available in your PATH."
    )



def stop_ollama(OLLAMA_HOST: str, process: subprocess.Popen):
    """
    Reliably stop the Ollama process started by this app.
    """
    if process is None:
        return
    # Already exited.
    if process.poll() is not None:
        return
    try:
        # Kill the entire process group.
        os.killpg(
            os.getpgid(process.pid),
            signal.SIGTERM,
        )
    except ProcessLookupError:
        return

    # Wait for Ollama to disappear.
    for _ in range(50):
        if not ollama_is_running(OLLAMA_HOST, verbose=True):
            return
        time.sleep(0.1)

    # Still alive -> force kill.
    try:
        os.killpg(
            os.getpgid(process.pid),
            signal.SIGKILL,
        )
    except ProcessLookupError:
        pass

    # Verify.
    for _ in range(20):
        if not ollama_is_running(OLLAMA_HOST):
            print("Ollama force-stopped.")
            return

        time.sleep(0.1)

    raise RuntimeError(
        "Could not stop Ollama. "
        "An independently started Ollama process may still be running."
    )


def check_ollama(OLLAMA_HOST):
    """
    Check whether Ollama is reachable.

    This makes debugging much easier than discovering later
    that Python cannot connect to localhost:11434.
    """

    print("Checking Ollama...")
    try:
        response = requests.get(
            f"{OLLAMA_HOST}/api/tags", timeout=10)

        response.raise_for_status()

    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Could not connect to Ollama at "
            f"{OLLAMA_HOST}\n\n"
            "Make sure Ollama is running.\n"
            "For example:\n"
            "ollama serve"
        )

    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            f"Ollama is reachable, but the request failed:\n{e}"
        )

    print("✓ Ollama is running")
    return True


# ============================================================
# CHECK THAT THE MODEL EXISTS
# ============================================================

def check_model(OLLAMA_HOST: str, model_name: str):
    """
    Verify that the requested model is installed.
    """

    print(f"Checking model: {model_name}")
    try:
        response = requests.get(
            f"{OLLAMA_HOST}/api/tags", timeout=10)

        response.raise_for_status()
        data = response.json()

    except Exception as e:
        raise RuntimeError(
            f"Could not retrieve Ollama model list: {e}"
        )

    models = data.get("models", [])

    installed_models = [
        model.get("name", "")
        for model in models
    ]

    if model_name not in installed_models:
        print("\nInstalled models:")

        for name in installed_models:
            print(f"  - {name}")

        raise RuntimeError(
            f"\nModel '{model_name}' is not installed.\n\n"
            f"Install it with:\n"
            f"   ollama pull {model_name}"
        )

    print(f"✓ Model '{model_name}' is installed")
    return True


# ============================================================
# IMAGE → BASE64
# ============================================================

def image_to_base64(image_path: str) -> str:
    """
    Read an image from disk and convert it to a base64 string.

    Ollama expects the image inside the 'images' field as
    base64-encoded data.
    """

    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    return image_base64


# ============================================================
# SEND IMAGE TO GLM-OCR
# ============================================================

def ocr_image(
        image_path: str, 
        model_name: str,
        OCR_PROMPT: str,
        TEMPERATURE: float,
        MAX_TOKENS: int,
        OLLAMA_URL: str,
        REQUEST_TIMEOUT: int
        ) -> str:
    """
    Send one image to GLM-OCR through Ollama using streaming.
    Detects repeating lines caused by GLM-OCR's missing EOS token and stops early.
    """
    print(f"Sending image to Ollama: {image_path}")
    image_base64 = image_to_base64(image_path)

    payload = {
        "model": model_name,
        "prompt": OCR_PROMPT,
        "images": [image_base64],
        "stream": True,  # Enable streaming to catch the repetition loop live
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": MAX_TOKENS,
            "num_ctx": 16384
        }
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=REQUEST_TIMEOUT,
            stream=True,
        )
        #response.raise_for_status()

        if not response.ok:
            detail = response.text
            response.close()
            raise RuntimeError(
                f"Ollama returned HTTP {response.status_code}: {detail}"
            )
        
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"HTTP request failed: {e}")

    extracted_lines = []
    seen_line_counts = {}
    MAX_ALLOWED_REPETITIONS = 3  # Stop if the exact same line repeats 3 times

    for line in response.iter_lines():
        if not line:
            continue
        
        try:
            chunk = json.loads(line.decode("utf-8"))
            token = chunk.get("response", "")

        except json.JSONDecodeError:
            continue

        # Accumulate tokens into text
        if token:
            # Append token to current text block
            if not extracted_lines:
                extracted_lines.append("")
            
            extracted_lines[-1] += token

            # If a newline occurs, evaluate the finished line for repeating patterns
            if "\n" in token:
                last_completed_line = extracted_lines[-1].strip()
                extracted_lines.append("")  # Start a new line buffer

                if last_completed_line:
                    seen_line_counts[last_completed_line] = seen_line_counts.get(last_completed_line, 0) + 1
                    
                    # If GLM-OCR gets stuck in an infinite repetition loop, abort generation manually
                    if seen_line_counts[last_completed_line] >= MAX_ALLOWED_REPETITIONS:
                        print(f"\n[INFO] Repetition loop detected for line: '{last_completed_line[:30]}...' - Halting generation.")
                        response.close()  # Force close the HTTP stream
                        break

    final_text = "".join(extracted_lines).strip()

    if not final_text:
        raise RuntimeError("GLM-OCR returned an empty response.")

    return final_text


# ============================================================
# PDF → MARKDOWN
# ============================================================

def parse_pdf_with_ollama(
    pdf_path: str,
    output_md_path: str,
    first_page_only: bool,
    PDF_DPI: int,
    OLLAMA_MODEL: str,
    OCR_PROMPT: str,
    OCR_TEMPERATURE: float,
    OCR_MAX_TOKENS: int,
    OCR_OLLAMA_URL: str,
    OCR_REQUEST_TIMEOUT: int
):
    """
    Convert PDF pages to images and OCR each page with GLM-OCR.

    Parameters
    ----------
    pdf_path:
        Path to input PDF.

    output_md_path:
        Path where Markdown will be written.

    first_page_only:
        True  = test only first page.
        False = process entire PDF.
    """

    # Validate PDF path, Ollama, and model.
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"PDF file not found:\n{pdf_path}"
        )

    check_model(OLLAMA_MODEL)


    print("\nConverting PDF to images...")
    pages = convert_from_path(
        pdf_path,
        dpi=PDF_DPI, # Resolution.
        fmt="png", # memento, PNG is lossless and works well for OCR.
        thread_count=4         # Convert several pages in parallel.
    )

    print(f"✓ PDF contains {len(pages)} page(s)")

    all_markdown = []

    # process each page
    for page_number, page in enumerate(pages, start=1):

        print(f"\n{'=' * 60}")

        print(
            f"Processing page "
            f"{page_number}/{len(pages)}")

        print(f"{'=' * 60}")

        # Create temporary image filename.
        temp_dir = "./temp_imgs"
        os.makedirs(temp_dir, exist_ok=True)
        temp_image = (f"./temp_imgs/glm_ocr_page_{page_number}.png")

        try:
            page.save(temp_image, format="PNG")

            print(
                f"✓ Saved page image: "
                f"{temp_image}"
            )


            markdown = ocr_image(
                temp_image,
                OLLAMA_MODEL,
                OCR_PROMPT,
                OCR_TEMPERATURE,
                OCR_MAX_TOKENS,
                OCR_OLLAMA_URL,
                OCR_REQUEST_TIMEOUT
            )

            print(f"\n--- PAGE {page_number} ---\n")

            print(markdown)

            print(f"\n--- END PAGE {page_number} ---")

            # Add page heading.
            page_markdown = (
                f"<!-- Page {page_number} -->\n\n"
                f"{markdown}")

            all_markdown.append(page_markdown)

        except Exception as e:

            # ------------------------------------------------
            # Don't kill the entire PDF if one page fails.
            # ------------------------------------------------

            print(
                f"\nERROR processing page "
                f"{page_number}:"
            )

            print(str(e))

            all_markdown.append(
                f"<!-- ERROR ON PAGE "
                f"{page_number}: {e} -->"
            )

        finally:
            # remove temporary image.
            if os.path.exists(temp_image):
                os.remove(temp_image)
                print(f"✓ Removed temporary image")

        # ----------------------------------------------------
        # TEST MODE
        # Set first_page_only=True to stop after page 1.
        # ----------------------------------------------------
        if first_page_only:
            print("\nFirst-page test complete.")
            break

    # ========================================================
    # COMBINE MARKDOWN
    # ========================================================

    print("\nCombining Markdown...")

    combined_markdown = (
        "\n\n---\n\n"
        .join(all_markdown)
    )

    # ========================================================
    # WRITE OUTPUT FILE
    # ========================================================

    with open(
        output_md_path,
        "w",
        encoding="utf-8"
    ) as output_file:

        output_file.write(
            combined_markdown
        )

    print(f"\n✓ Extraction complete.")

    print(
        f"✓ Output written to:\n"
        f"{output_md_path}")


## --- SHARED UTILS ---

def format_timestamp(seconds) -> str:
    """Helper function to format seconds into MM:SS format."""
    try:
        sec_float = float(seconds)
        mins = int(sec_float // 60)
        secs = int(sec_float % 60)
        return f"{mins:02d}:{secs:02d}"
    except (ValueError, TypeError):
        return "00:00"

def load_prompts(filepath: str) -> dict:
    """Loads system prompts. Creates default Obsidian and Quiz prompts if missing."""

    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                prompts = json.load(f)
                return prompts
        except Exception:
            pass
    # If file doesn't exist or is corrupted, return an error
    st.warning("⚠️ System prompts file missing or unreadable. Please fill it with valid JSON prompts.")
    return {}

def save_prompt(filepath: str, name: str, prompt_text: str):
    """Saves a new or updated prompt to the JSON file."""
    prompts = {}
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            prompts = json.load(f)
    prompts[name] = prompt_text
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(prompts, f, indent=4)



# ============================================================
# chatting-related functions
# ============================================================
from pathlib import Path
from datetime import datetime

def save_chat_as_markdown(messages, chat_started_at, chat_filepath):
    """
    Save the current chat conversation as a Markdown file.
    """

    if not messages:
        return None

    if chat_started_at:
        timestamp = chat_started_at.strftime("%Y-%m-%d_%H-%M-%S")
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    lines = []

    lines.append("# Chat Conversation")
    lines.append("")
    lines.append(
        f"**Started:** "
        f"{chat_started_at.strftime('%Y-%m-%d %H:%M:%S') if chat_started_at else 'Unknown'}"
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    for message in messages:

        role = message["role"]
        content = message["content"]

        if role == "user":
            lines.append("## 👤 User")
        elif role == "assistant":
            lines.append("## 🤖 Assistant")
        elif role == "system":
            lines.append("## ⚙️ System")
        else:
            lines.append(f"## {role.title()}")

        lines.append("")
        lines.append(content)
        lines.append("")
        lines.append("---")
        lines.append("")

    chat_filepath = Path(chat_filepath)
    chat_filepath.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return chat_filepath

def stream_ollama_chat(
    messages: list,
    ollama_url: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    top_p: float,
    top_k: int,
    thinking=True,
):
    """
    Stream a chat response from Ollama.

    Yields dictionaries with one of these types:

        {"type": "thinking", "content": "..."}
        {"type": "content", "content": "..."}
        {"type": "done"}

    Thinking and final answer are deliberately separated.
    """

    # -----------------------------------------------------
    # Ollama generation options
    # -----------------------------------------------------

    options = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "num_predict": max_tokens,
    }

    # -----------------------------------------------------
    # Request payload
    # -----------------------------------------------------

    payload = {
        "model": model,
        "messages": messages,

        # We MUST stream from Ollama in order to receive
        # thinking chunks as they are generated.
        "stream": True,

        # Ask a thinking-capable model to produce thinking.
        #
        # Models that don't support thinking may simply
        # return no "thinking" field.
        "think": thinking,

        "options": options,
    }

    # -----------------------------------------------------
    # Open streaming HTTP connection
    # -----------------------------------------------------

    response = requests.post(
        ollama_url,
        json=payload,
        stream=True,
        timeout=timeout,
    )

    response.raise_for_status()

    # -----------------------------------------------------
    # Ollama returns newline-delimited JSON.
    #
    # Each line represents one partial response.
    # -----------------------------------------------------

    for line in response.iter_lines(decode_unicode=True):

        if not line:
            continue

        try:
            chunk = json.loads(line)

        except json.JSONDecodeError:
            # Ignore malformed/empty lines rather than
            # killing the entire conversation.
            continue

        # -------------------------------------------------
        # Ollama may return an error object.
        # -------------------------------------------------

        if "error" in chunk:
            raise RuntimeError(chunk["error"])

        message = chunk.get("message", {})

        # -------------------------------------------------
        # Thinking/reasoning chunk
        # -------------------------------------------------

        thinking_text = message.get("thinking")

        if thinking_text:
            yield {
                "type": "thinking",
                "content": thinking_text,
            }

        # -------------------------------------------------
        # Normal answer chunk
        # -------------------------------------------------

        content_text = message.get("content")

        if content_text:
            yield {
                "type": "content",
                "content": content_text,
            }

        # -------------------------------------------------
        # Final chunk
        # -------------------------------------------------

        if chunk.get("done"):
            yield {
                "type": "done",
            }

## --- YOUTUBE UTILS ---

from youtube_transcript_api import YouTubeTranscriptApi
import re

def extract_video_id(url: str) -> str:
    """Extracts the YouTube Video ID from any standard, shortened, live, or shorts URL."""
    # Strip any accidental spaces the user might have copied
    url = url.strip()
    
    # This Regex pattern hunts for the 11-character ID in almost any YouTube URL variation
    regex_pattern = r"(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/|youtube\.com\/shorts\/|youtube\.com\/live\/)([^\"&?\/\s]{11})"
    
    match = re.search(regex_pattern, url)
    
    if match:
        return match.group(1)
        
    st.error(f"❌ Error: Could not find a valid 11-character video ID in the URL: {url}")
    return ""

def fetch_youtube_metadata(video_url: str) -> dict:
    """Fetches video details via oEmbed and scrapes the HTML for the Upload Date."""
    meta_data = {
        "video_url": video_url, 
        "title": "Unknown Title", 
        "author_name": "Unknown Channel", 
        "author_url": "", 
        "thumbnail_url": "",
        "upload_date": "Unknown Date"
    }
    
    try:
        # 1. Get Title and Author safely from oEmbed
        oembed_url = f"https://www.youtube.com/oembed?url={video_url}&format=json"
        resp = requests.get(oembed_url, timeout=5)
        if resp.status_code == 200:
            meta = resp.json()
            meta_data["title"] = meta.get("title", "Unknown Title")
            meta_data["author_name"] = meta.get("author_name", "Unknown Channel")
            meta_data["author_url"] = meta.get("author_url", "")
            meta_data["thumbnail_url"] = meta.get("thumbnail_url", "")
        # 2. Scrape the raw YouTube page for the exact upload date
        html_resp = requests.get(video_url, timeout=5)
        
        # Regex hunts for: <meta itemprop="uploadDate" content="2023-10-25T...">
        date_match = re.search(r'<meta itemprop="uploadDate" content="(.*?)">', html_resp.text)
        if date_match:
            # Split by "T" to just get the YYYY-MM-DD part
            meta_data["upload_date"] = date_match.group(1).split("T")[0] 
            
    except Exception as e:
        st.warning(f"Could not fetch complete metadata: {e}")
    
    return meta_data

def fetch_transcript_with_logs(url: str, video_id: str) -> dict:
    """
    Fetches transcript using the free youtube-transcript-api 
    and grabs metadata using YouTube's free oEmbed endpoint.
    """
    st.write(f"🔍 Starting extraction for video ID: `{video_id}`...")
    
    # --- 1. Fetch Metadata (oEmbed) ---
    st.write("📊 Fetching video metadata (title, author)...")
    metadata = {
        "title": "Unknown Title",
        "author_name": "Unknown Channel",
        "video_url": url,
        "upload_date": "Unknown"
    }
    
    try:
        oembed_url = f"https://www.youtube.com/oembed?url={url}&format=json"
        response = requests.get(oembed_url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            metadata["title"] = data.get("title", metadata["title"])
            metadata["author_name"] = data.get("author_name", metadata["author_name"])
            st.write(f"✅ Found video: **{metadata['title']}**")
        else:
            st.warning("⚠️ Could not fetch complete metadata, proceeding with defaults.")
    except Exception as e:
        st.warning(f"⚠️ Metadata fetch error: {e}")

    # --- 2. Fetch Transcript ---
    st.write("🌐 Calling free YouTube Transcript API...")
    try:
        yt_api = YouTubeTranscriptApi()
        raw_transcript = yt_api.fetch(video_id)
        
        segments = []
        for item in raw_transcript:
            segments.append({
                # CHANGED: We now access these as object attributes instead of dict keys
                "start": getattr(item, "start", 0.0), 
                "text": getattr(item, "text", "")
            })
        
        st.success(f"✅ Successfully extracted {len(segments)} transcript segments!")
        return {
            "segments": segments,
            "metadata": metadata
        }
        
    except Exception as e:
        # Pushing the exact error to the UI so the user can read it
        st.error(f"❌ Transcript extraction failed! Error: {e}")
        st.info("💡 Hint: This usually happens if the video has subtitles disabled by the creator, is age-restricted, or is a live stream.")
        return {}
    

def format_transcript_for_copy(transcript_data: list) -> str:
    """Compiles the edited transcript segments into a single, clean text block."""
    return "\n\n".join([segment.get('text', '') for segment in transcript_data])

def get_cached_videos(cache_dir: str) -> dict:
    """
    Scans the cache directory for saved video JSON files.
    Returns a dictionary mapping 'video_id' to its 'title'.
    This is used to populate the dropdown menu in the UI.
    """
    cached_videos = {}
    
    if not os.path.exists(cache_dir):
        return cached_videos
        
    for filename in os.listdir(cache_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(cache_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                # Extract the video ID from the filename (remove .json)
                video_id = filename.replace(".json", "")
                
                # Try to get the title from metadata, default to the ID if not found
                title = data.get("metadata", {}).get("title", f"Unknown Title ({video_id})")
                
                cached_videos[video_id] = title
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                
    return cached_videos

def load_cached_video(cache_dir: str, video_id: str) -> dict:
    """
    Reads a specific video's JSON file from disk and returns the dictionary.
    """
    filepath = os.path.join(cache_dir, f"{video_id}.json")
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_edits_to_disk(CACHE_DIR: str):
    """Saves the current session state transcript back to the local JSON file."""
    if not st.session_state.get("video_id"):
        return

    cache_filepath = os.path.join(CACHE_DIR, f"{st.session_state.video_id}.json")
    
    # Reconstruct the full data dictionary
    full_data = {
        "metadata": st.session_state.metadata,
        "segments": st.session_state.transcript
    }
    
    try:
        with open(cache_filepath, "w", encoding="utf-8") as f:
            json.dump(full_data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        st.error(f"⚠️ Failed to save changes: {e}")



## --- RAG UTILS ---

import hashlib

def render_chunk_ui(
        doc: str, 
        score: float, 
        meta: str, 
        citation: str, 
        filename: str, 
        c_idx: int):
    # Generate a universally unique ID based on the chunk's actual text content
    unique_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()[:8]
    txt_key = f"chunk_txt_{unique_hash}_{c_idx}"
    chk_key = f"chk_apprv_{unique_hash}_{c_idx}"
    
    col_chunk, col_meta = st.columns([3, 1])
    
    with col_chunk:
        score_label = "Vector Relevance"
        st.caption(f"**Chunk #{c_idx + 1}** | {score_label}: `{score:.4f}`")
        
        edited_chunk = st.text_area(
            "Edit this chunk before final context insertion:", 
            value=doc.page_content, 
            height=150, 
            key=txt_key,
            label_visibility="collapsed"
        )
        
        is_approved = st.checkbox(f"✅ Approve Chunk #{c_idx + 1} for Final Synthesis", key=chk_key)
        
    with col_meta:
        st.info("**Extracted Citation:**")
        st.markdown(citation)
        with st.expander("View Raw metadata.json"):
            st.json(meta)

    # Save to State if Approved
    if is_approved:
        st.session_state.curated_contexts[c_idx] = {
            "text": edited_chunk,
            "citation": citation,
            "source": filename
        }
    else:
        st.session_state.curated_contexts.pop(c_idx, None)
