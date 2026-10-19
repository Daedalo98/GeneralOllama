import os
import streamlit as st

# ==========================================
# SYSTEM LOGGER INITIALIZATION AND GLOBAL STATES
# ==========================================

if "folder_paths" not in st.session_state:
    st.session_state.folder_paths = {}

if "app_logs" not in st.session_state:
    st.session_state.app_logs = []
if "global_zoom" not in st.session_state:
    st.session_state.global_zoom = 22 # Default font size in pixels

def sync_zoom(slider_key):
    """Callback to update the global zoom whenever ANY of the local sliders are moved."""
    for slider in ["zoom_step1", "zoom_step2"]:
        if slider != slider_key:
            st.session_state[slider] = st.session_state[slider_key]
            st.session_state.global_zoom = st.session_state[slider_key]

if "thinking_text" not in st.session_state:
    st.session_state.thinking_text = ""
    
if "edit_text" not in st.session_state:
    st.session_state.edit_text = False

if "prompts_dict" not in st.session_state:
    st.session_state.prompts_dict = {}

if "ollama_process" not in st.session_state:
    st.session_state.ollama_process = None

if "ollama_started_by_app" not in st.session_state:
    st.session_state.ollama_started_by_app = False

if "context_length" not in st.session_state:
    st.session_state.context_length = 4096

if "ollama_models" not in st.session_state:
    st.session_state.ollama_models = []

if "ollama_host" not in st.session_state:
    st.session_state.ollama_host = ""


# =========================================================
# PARSING STATE
# =========================================================

if "is_parsing" not in st.session_state:
    st.session_state.is_parsing = False

# =========================================================
# QUIZZING STATE
# =========================================================

if "is_quizzing" not in st.session_state:
    st.session_state.is_quizzing = False

# =========================================================
# CHAT SESSION STATE
# =========================================================

if "chat_start" not in st.session_state:
    st.session_state.chat_start = False

if "chat_active" not in st.session_state:
    st.session_state.chat_active = False

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "chat_started_at" not in st.session_state:
    st.session_state.chat_started_at = None

# IMPORTANT:
# Store a snapshot of the system prompt used by this chat.
#
# We do NOT want the saved chat to depend on whichever
# prompt happens to be selected later in the sidebar.
if "chat_system_prompt" not in st.session_state:
    st.session_state.chat_system_prompt = ""

# Optional but useful: remember which model started the chat.
if "chat_model" not in st.session_state:
    st.session_state.chat_model = ""



# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.subheader("Global Zoom")
    st.slider("Adjust Global Zoom", min_value=10, max_value=40, value=st.session_state.global_zoom, key="zoom_step1", on_change=sync_zoom, args=("zoom_step1",))

    # ---------------------------------------------------------
    # System Logs
    # ---------------------------------------------------------
    st.divider()
    st.subheader("📝 System Logs")
    with st.container(height=300, border=True):
        if not st.session_state.app_logs:
            st.caption("No logs yet...")
        else:
            for log in st.session_state.app_logs:
                st.text(log) # st.text keeps the formatting clean


    # =========================================================
    # FOLDERS AND PATHS SETTINGS
    # =========================================================
    st.markdown("# Folders and Paths management")

    with st.container(border=False):
        with st.expander("📁 Folder paths management", expanded=False):
            # pdf folder
            PDF_FOLDER = st.text_input("PDF Files folder", value="./pdfs")
            os.makedirs(PDF_FOLDER, exist_ok=True)
            st.session_state.folder_paths["PDF_FOLDER"] = PDF_FOLDER

            # md folder
            MD_FOLDER = st.text_input("MD Files Folder", value="./mds")
            os.makedirs(MD_FOLDER, exist_ok=True)
            st.session_state.folder_paths["MD_FOLDER"] = MD_FOLDER

            # database folder
            DB_FOLDER = st.text_input("vector DATABASE Folder", value="./chroma_db")
            os.makedirs(DB_FOLDER, exist_ok=True)
            st.session_state.folder_paths["DB_FOLDER"] = DB_FOLDER

            # chat folder
            CHAT_FOLDER = st.text_input("CHAT storage folder", value=f"./mds/chats/")
            os.makedirs(CHAT_FOLDER, exist_ok=True)
            st.session_state.folder_paths["CHAT_FOLDER"] = CHAT_FOLDER
            
            # video folder
            VIDEO_FOLDER = st.text_input("VIDEO notes folder", value=f"./mds/videos/")
            os.makedirs(VIDEO_FOLDER, exist_ok=True)
            st.session_state.folder_paths["VIDEO_FOLDER"] = VIDEO_FOLDER
                

# ==========================================
# MAGIC CSS INJECTION (Globally Scaled)
st.markdown(f"""
    <style>
        .stTextArea textarea {{ font-size: {st.session_state.global_zoom}px !important; line-height: 1.5 !important; }}
        .stMarkdown p, .stMarkdown li {{ font-size: {st.session_state.global_zoom}px !important; line-height: 1.5 !important; }}
        .stRadio p, .stCheckbox p, .stAlert p {{ font-size: {st.session_state.global_zoom}px !important; line-height: 1.5 !important; }}
    </style>""", 
    unsafe_allow_html=True)
# ==========================================


# ==========================================
# MULTI-PAGE SETUP
# ==========================================
pg = st.navigation([
    st.Page("./pages/welcome.py", title="Welcome", icon="👋"),
    st.Page("./pages/settings.py", title="Settings", icon="⚙️"),
    st.Page("./pages/ollamachat.py", title="Ollama Chat", icon="🦙"), 
    st.Page("./pages/youtube.py", title="YouTube", icon="🎞️"),
    st.Page("./pages/zotero.py", title="Zotero", icon="📜"),
    st.Page("./pages/quiz.py", title="Quiz Time", icon="❓"),
    st.Page("./pages/parsing.py", title="Parsing", icon="📖"),
    st.Page("./pages/rag.py", title="RAG", icon="🛸"),
    ])
pg.run()


