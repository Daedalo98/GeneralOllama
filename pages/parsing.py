from __future__ import annotations

import json
import time
import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from pypdf import PdfReader

import functions
import requests

def path_key(path: Path) -> str:
    """Return a consistent string key for paths used in session state."""
    return str(path.resolve())

def selection_file_for(source_dir: Path) -> Path:
    """Store the selection in the source folder, keyed to that folder."""
    return source_dir / SELECTION_FILENAME

def get_page_count(pdf_path: Path) -> tuple[int | None, str]:
    """Return the PDF page count, or an explanatory error status."""
    try:
        reader = PdfReader(str(pdf_path), strict=False)

        if reader.is_encrypted:
            try:
                result = reader.decrypt("")
                if result == 0:
                    return None, "Encrypted PDF"
            except Exception:
                return None, "Encrypted PDF"

        return len(reader.pages), "OK"
    except Exception as exc:
        return None, f"Could not read: {type(exc).__name__}"

def build_markdown_index(output_dir: Path) -> dict[str, list[Path]]:
    """Index Markdown files by their filename stem, case-insensitively."""
    index: dict[str, list[Path]] = {}

    if not output_dir.is_dir():
        return index

    for md_path in output_dir.rglob("*.md"):
        index.setdefault(md_path.stem.casefold(), []).append(md_path)

    return index

def find_markdown_match(
    pdf_path: Path,
    source_dir: Path,
    output_dir: Path,
    markdown_index: dict[str, list[Path]],
) -> tuple[Path | None, str]:
    """
    Match a PDF to Markdown.

    Preferred layout preserves relative paths:
        source/reports/a.pdf -> output/reports/a.md

    If that file is absent, a flat filename match is accepted only when unique.
    """
    relative_pdf = pdf_path.relative_to(source_dir)
    expected_md = (output_dir / relative_pdf).with_suffix(".md")

    if expected_md.is_file():
        return expected_md, "Parsed"

    candidates = markdown_index.get(pdf_path.stem.casefold(), [])

    if len(candidates) == 1:
        return candidates[0], "Parsed (filename match)"

    if len(candidates) > 1:
        return None, "Ambiguous Markdown match"

    return None, "Not parsed"

def make_inventory(
    source_dir: Path,
    output_dir: Path,
) -> pd.DataFrame:
    
    """Scan PDFs and return one inventory row per PDF."""
    pdf_paths = functions.list_of_filetype(source_dir = source_dir, filetype="pdf")
    markdown_index = build_markdown_index(output_dir)
    rows: list[dict[str, Any]] = []

    for pdf_path in pdf_paths:
        relative_path = pdf_path.relative_to(source_dir)
        page_count, page_status = get_page_count(pdf_path)
        markdown_path, match_status = find_markdown_match(
            pdf_path, source_dir, output_dir, markdown_index
        )

        # If the source is newer than the matching Markdown, flag it as stale.
        if markdown_path and match_status.startswith("Parsed"):
            try:
                if pdf_path.stat().st_mtime > markdown_path.stat().st_mtime:
                    match_status = "Parsed — source PDF is newer"
            except OSError:
                pass

        rows.append(
            {
                "Select": False,  # Filled from saved/session selection below.
                "File name": pdf_path.name,
                "Relative path": str(relative_path),
                "Pages": page_count,
                "PDF status": page_status,
                "Parse status": match_status,
                "Markdown path": (
                    str(markdown_path.relative_to(output_dir))
                    if markdown_path
                    else ""
                ),
                "_key": path_key(pdf_path),
            }
        )

    return pd.DataFrame(rows)

def get_selected_pdfs(file_or_list, pdf_folder, isfile):
    """
    Read the inventory JSON and return selected PDF paths.

    The inventory stores paths relative to pdf_folder. The resolved-path
    check prevents entries in the JSON from pointing outside that folder.
    """

    if isfile: 
        records = json.load(file_or_list)
    else:
        records = file_or_list

    source_root = Path(pdf_folder).expanduser().resolve()

    selected_paths = []

    for record in records:
        if not record.get("selected", False):
            continue

        relative_path = Path(record["pdf_path"])

        # Inventory paths should be relative, never absolute.
        if relative_path.is_absolute():
            raise ValueError(f"Expected a relative PDF path: {relative_path}")

        pdf_path = (source_root / relative_path).resolve()

        # Reject paths that escape the configured PDF folder.
        if not pdf_path.is_relative_to(source_root):
            raise ValueError(f"PDF path is outside the source folder: {relative_path}")

        if not pdf_path.is_file():
            st.warning(f"Selected PDF no longer exists: {pdf_path}")
            continue

        selected_paths.append(pdf_path)

    return selected_paths


# main parameters
prompt_names = list(st.session_state.prompts_dict.keys())
ollama_models = st.session_state.ollama_models
OLLAMA_HOST = st.session_state.ollama_host
SELECTION_FILENAME = ".pdf_inventory_selection.json"

PDF_FOLDER = st.session_state.folder_paths["PDF_FOLDER"]
source_dir = Path(PDF_FOLDER).expanduser()
selection_file = selection_file_for(source_dir)

MD_FOLDER = st.session_state.folder_paths["MD_FOLDER"]
output_dir = Path(MD_FOLDER).expanduser()

with st.sidebar:
    # ---------------------------------------------------------
    # OCR settings
    # ---------------------------------------------------------
    with st.container(border=False):             
        with st.expander("🧠 OCR model settings", expanded=True):
            ocr_idx = ollama_models.index("glm-ocr:bf16") if "glm-ocr:bf16" in ollama_models else 0
            OCR_MODEL = st.selectbox("OCR Model", ollama_models, index=ocr_idx)

            with st.expander("Advanced Settings", expanded=False, icon="⚙️"):
                # Ollama's local HTTP API.
                OCR_OLLAMA_URL = st.text_input("OCR Ollama URL", value=f"{OLLAMA_HOST}/api/generate")

                # PDF resolution.
                #
                # For small text you may want 250-300 DPI.
                # Higher DPI = larger images = more memory and potentially
                # slower processing.
                OCR_PDF_DPI = st.number_input("PDF DPI", 50, 500, 200, 10, help="Higher DPI = better OCR accuracy but slower processing and more memory usage.")

                # Generation temperature.
                # OCR should generally be deterministic, so 0 is appropriate.
                OCR_TEMPERATURE = st.number_input("OCR Temperature", 0.0, 1.0, 0.0, 0.1, help="Higher values make output more random. For OCR, 0 is recommended for deterministic output.")

                # Maximum number of generated tokens.
                OCR_MAX_TOKENS = st.number_input("OCR Max Tokens", -1, 100000, 10000, 1, help="Maximum number of tokens to generate.")

                # HTTP timeout.
                #
                # IMPORTANT:
                # OCR models can take longer than 30 seconds, especially when
                # running locally on CPU.
                OCR_REQUEST_TIMEOUT = st.number_input("OCR Request Timeout", 1, 1000, 600, 1, help="Timeout for OCR requests in seconds.")

                st.subheader("System Prompts")
                default_index = (
                    prompt_names.index("OCR_PROMPT")
                    if "OCR_PROMPT" in prompt_names else 0)

                ocr_prompt_name = st.selectbox(
                    "Active Prompt",
                    options=prompt_names,
                    index=default_index,
                    key="ocr_prompt_selector"
                )

                ocr_prompt_text = st.session_state.prompts_dict.get(ocr_prompt_name,"")
                # show the prompt
                st.text(ocr_prompt_text)

            if st.session_state.context_length < 96000:
                st.info("Restart Ollama with a bigger context (try 96k or more) to parse documents.")


st.markdown("# PDF Inventory")
st.write(
    "Review PDFs and page counts, compare files with Markdown outputs, "
    "and choose which files should be included in a future parsing run."
)

# main button to rescan folder after some changes
scan_button = st.button("Rescan folders", type="primary", width="stretch")

if not source_dir.is_dir():
    st.error(f"Source folder does not exist or is not a folder: {source_dir}")
    st.stop()

if not output_dir.is_dir():
    st.info(f"Markdown output folder does not exist yet: {output_dir}")

# Scan initially and when the user explicitly requests a rescan.
if "inventory" not in st.session_state or scan_button:
    with st.spinner("Scanning PDFs and counting pages…"):
        st.session_state.inventory = make_inventory(
            source_dir,
            output_dir
        )
    st.session_state.inventory_source = path_key(source_dir)

inventory = st.session_state.inventory.copy()

# If the source folder changed, discard the previous scan and selection state.
if st.session_state.get("inventory_source") != path_key(source_dir):
    with st.spinner("Scanning the selected source folder…"):
        inventory = make_inventory(
            source_dir,
            output_dir
        )
    st.session_state.inventory = inventory
    st.session_state.inventory_source = path_key(source_dir)

saved_selection = functions.load_json_as_booldict(selection_file)
session_selection = st.session_state.get("file_selection", {})
selection: dict[str, bool] = {**saved_selection, **session_selection}

if not inventory.empty:
    for _, row in inventory.iterrows():
        key = row["_key"]
        if key not in selection:
            # Default to selecting files that do not appear to be parsed.
            selection[key] = not str(row["Parse status"]).startswith("Parsed")

    inventory["Select"] = inventory["_key"].map(selection).fillna(False).astype(bool)

# Summary metrics refer to the full inventory, not the filtered table.
pdf_count = len(inventory)
total_pages = int(inventory["Pages"].dropna().sum()) if pdf_count else 0
parsed_count = (
    int(inventory["Parse status"].str.startswith("Parsed").sum())
    if pdf_count
    else 0
)
unparsed_count = pdf_count - parsed_count
selected_count = int(inventory["Select"].sum()) if pdf_count else 0
selected_pages = (
    int(inventory.loc[inventory["Select"], "Pages"].dropna().sum())
    if pdf_count
    else 0
)

metric_cols = st.columns(5)
metric_cols[0].metric("PDFs found", pdf_count)
metric_cols[1].metric("Total pages", total_pages)
metric_cols[2].metric("Parsed", parsed_count)
metric_cols[3].metric("Not parsed / ambiguous", unparsed_count)
metric_cols[4].metric("Selected", f"{selected_count} · {selected_pages} pages")

if pdf_count == 0:
    st.info("No PDFs were found with the current folder and depth settings.")
    st.stop()

# Bulk selection controls.
button_cols = st.columns(3)

if button_cols[0].button("Select all unparsed"):
    for _, row in inventory.iterrows():
        if not str(row["Parse status"]).startswith("Parsed"):
            selection[row["_key"]] = True

if button_cols[1].button("Select all"):
    for key in inventory["_key"]:
        selection[key] = True

if button_cols[2].button("Select none"):
    for key in inventory["_key"]:
        selection[key] = False

inventory["Select"] = inventory["_key"].map(selection).fillna(False).astype(bool)

# Filters apply to what is shown, not to the inventory or summary metrics.
filter_cols = st.columns([2, 2, 3])
status_filter = filter_cols[0].selectbox(
    "Parse status",
    ["All", "Not parsed", "Parsed", "Ambiguous match", "Source newer"],
)
selection_filter = filter_cols[1].selectbox(
    "Selection",
    ["All", "Selected", "Not selected"],
)
search_text = filter_cols[2].text_input("Search filename or path")

visible = inventory.copy()

if status_filter == "Not parsed":
    visible = visible[visible["Parse status"] == "Not parsed"]
elif status_filter == "Parsed":
    visible = visible[visible["Parse status"].str.startswith("Parsed")]
elif status_filter == "Ambiguous match":
    visible = visible[visible["Parse status"] == "Ambiguous Markdown match"]
elif status_filter == "Source newer":
    visible = visible[visible["Parse status"].str.contains("source PDF is newer")]

if selection_filter == "Selected":
    visible = visible[visible["Select"]]
elif selection_filter == "Not selected":
    visible = visible[~visible["Select"]]

if search_text.strip():
    term = search_text.strip().casefold()
    visible = visible[
        visible["File name"].str.casefold().str.contains(term, na=False)
        | visible["Relative path"].str.casefold().str.contains(term, na=False)
    ]

st.caption(
    "Select or clear the checkbox in the table to change the queue. "
    "Selections are saved in the source folder."
)

display_columns = [
    "Select",
    "File name",
    "Relative path",
    "Pages",
    "PDF status",
    "Parse status",
    "Markdown path",
]

edited = st.data_editor(
    visible[display_columns + ["_key"]],
    hide_index=True,
    width="stretch",
    disabled=[column for column in display_columns if column != "Select"]
    + ["_key"],
    column_config={
        "Select": st.column_config.CheckboxColumn(
            "Select for parsing",
            help="This only controls selection; it does not start parsing.",
        ),
        "Pages": st.column_config.NumberColumn("Pages", format="%d"),
        "_key": None,
    },
    key="inventory_table",
)

# Apply edits to the complete selection map, including rows hidden by filters.
for _, row in edited.iterrows():
    selection[row["_key"]] = bool(row["Select"])

st.session_state.file_selection = selection
functions.save_booldict_as_json(selection_file, selection)

selected_rows = inventory[inventory["_key"].map(selection).fillna(False)]
st.subheader("Selected files")
st.write(f"{len(selected_rows)} files · {int(selected_rows['Pages'].dropna().sum())} pages")

if selected_rows.empty:
    st.write("No files are selected.")
    st.write("Import JSON file or select rows to start parsing.")

else:
    st.dataframe(
        selected_rows[["File name", "Relative path", "Pages", "Parse status"]],
        hide_index=True,
        width="stretch",
    )

    # Export the selection for use by a separate parsing script or app.
    export_rows = [
        {
            "pdf_path": row["Relative path"],
            "selected": bool(selection.get(row["_key"], False)),
            "pages": row["Pages"] if pd.notna(row["Pages"]) else None,
            "parse_status": row["Parse status"],
            "markdown_path": row["Markdown path"],
        }
        for _, row in inventory.iterrows()
    ]

    col1, col2 = st.columns([1.5, 1], gap="small", border=False, width="stretch")
    with col1:
        st.download_button(
            "Download inventory and selection as JSON for future parsing",
            data=json.dumps(export_rows, indent=2),
            file_name="pdf_inventory_selection_for_parsing.json",
            mime="application/json",
            help="download json file to parse selected PDFs in another session."
        )
    with col2:
        if st.button("Upload JSON file"):
            selection_upload = st.file_uploader(
                "Upload the inventory selection JSON",
                type=["json"],
                help="Use the JSON downloaded from the PDF inventory app.",
            )
        else:
            selection_upload = None

# =========================================================
# PARSING MAIN SECTION 
# =========================================================
if selection_upload is None:
    try:
        selected_pdfs = get_selected_pdfs(export_rows, PDF_FOLDER, isfile=False)
    except Exception as e:
        st.warning(f"no file uploaded and no selected pdf, error: {e}")

else:
    try:
        selected_pdfs = get_selected_pdfs(selection_upload, PDF_FOLDER, isfile=True)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        st.error(f"Could not read the selection file: {exc}")
        selected_pdfs = []

st.write(f"Selected PDFs: {len(selected_pdfs)}")

for pdf_path in selected_pdfs:
    st.write(pdf_path.relative_to(Path(PDF_FOLDER).expanduser().resolve()))

if st.session_state.context_length < 96000:
    st.warning("Parsing require a big context lenght for optimal results. min 112k is suggested.")

if st.button(
    "Start parsing",
    disabled=(
        st.session_state.context_length < 96000
        or not selected_pdfs
    ),
    help="This takes time and may use most of your compute resources.",
    type="primary",
    width="stretch"
):
    if functions.check_ollama(OLLAMA_HOST) and functions.check_model(OLLAMA_HOST, OCR_MODEL):
        st.session_state.is_parsing = True

        # Save paths in session state. Streamlit reruns the script after
        # interactions, so don't rely on a local variable from a previous run.
        st.session_state.parse_queue = [str(path) for path in selected_pdfs]

        functions.add_log(f"selected {len(selected_pdfs)} PDF file(s)")
        functions.add_log("parsing process started.")

        st.session_state.is_parsing = True

        start_time = time.time()
        parse_queue = st.session_state.get("parse_queue", [])

        pdf_progress = st.progress(
            value=0,
            text=f"Parsing {len(parse_queue)} selected PDF file(s)",
        )

        

if st.session_state.is_parsing:
    for idx, filepath in enumerate(parse_queue, start=1):
        filepath = Path(filepath)

        # ---- Process the file ----
        pages = functions.convert_from_path(
            str(filepath),
            dpi=OCR_PDF_DPI,
            fmt="png",
            thread_count=4,
        )

        all_markdown = []
        source_root = Path(PDF_FOLDER).expanduser().resolve()
        output_root = Path(MD_FOLDER).expanduser().resolve()

        # Keep the PDF's relative folder structure in the Markdown output folder.
        relative_pdf = filepath.resolve().relative_to(source_root)
        output_md_path = (output_root / relative_pdf).with_suffix(".md")

        # Create the matching output subfolder if needed.
        output_md_path.parent.mkdir(parents=True, exist_ok=True)

        # process each page
        page_progress = st.progress(value=0)
        page_start_time = time.time()
        for page_number, page in enumerate(pages, start=1):
            
            # Create temporary image filename.
            temp_dir = "./temp_imgs"
            os.makedirs(temp_dir, exist_ok=True)
            temp_image = (f"./temp_imgs/glm_ocr_page_{page_number}.png")
            page.save(temp_image, format="PNG")
            #print(f"✓ Saved page image: {temp_image}")

            try:
                markdown = functions.ocr_image(
                    temp_image,
                    OCR_MODEL,
                    ocr_prompt_text,
                    OCR_TEMPERATURE,
                    OCR_MAX_TOKENS,
                    OCR_OLLAMA_URL,
                    OCR_REQUEST_TIMEOUT
                )

                # Add page heading.
                page_markdown = (
                    f"<!-- Page {page_number} -->\n\n"
                    f"{markdown}")

                all_markdown.append(page_markdown)

            except Exception as e:
                # Don't kill the entire PDF if one page fails.
                st.warning(f"\nERROR processing page {page_number}: {str(e)}")
                functions.add_log(f"\nERROR processing page {page_number}: {str(e)}")

                all_markdown.append(
                    f"<!-- ERROR ON PAGE "
                    f"{page_number}: {e} -->"
                )

            finally:
                # remove temporary image.
                if os.path.exists(temp_image):
                    os.remove(temp_image)
                    #print(f"✓ Removed temporary image")

            page_elapsed = time.time() - page_start_time
            avg_time_per_page = page_elapsed / page_number
            page_remaining_time_s = avg_time_per_page * (len(pages) - page_number)

            # Whole days
            days = page_remaining_time_s // (60 * 60 * 24)
            secs = page_remaining_time_s % (60 * 60 * 24)

            # Hours, minutes, seconds
            hours = secs // (60 * 60)
            mins = (secs % (60 * 60)) // 60
            secs = secs % 60

            # Build the string; you can adjust the wording or zero‑padding as you like
            page_remaining_time_text = f"{days} days {hours} hours {mins} minutes {secs:.0f} seconds"
            page_progress.progress(int(page_number*100/len(pages)), text=f"Processed page {page_number}/{len(pages)} \n\nRemaining time: (approx) {page_remaining_time_text}")

        # WRITE OUTPUT FILE
        combined_markdown = ("\n\n---\n\n".join(all_markdown))
        with output_md_path.open("w", encoding="utf-8") as output_file:
            output_file.write(combined_markdown)

        # ---- Compute remaining time estimate ----
        pdf_elapsed = time.time() - start_time
        #print(elapsed)
        avg_time_per_file = pdf_elapsed / idx
        remaining_time_s = avg_time_per_file * (len(parse_queue) - idx)

        # Whole days
        days = remaining_time_s // (60 * 60 * 24)
        secs = remaining_time_s % (60 * 60 * 24)

        # Hours, minutes, seconds
        hours = secs // (60 * 60)
        mins = (secs % (60 * 60)) // 60
        secs = secs % 60

        # Build the string; you can adjust the wording or zero‑padding as you like
        remaining_time_text = f"{days} days {hours} hours {mins} minutes {secs} seconds"
        pdf_progress.progress(int(idx*100/len(parse_queue)), text=f"parsing file n.{idx} of {len(parse_queue)} files \n\n average remaining time: f{remaining_time_text}")

    # when the loop is done, the state "is parsing" is not True and the queue is empty
    st.session_state.is_parsing = False
    st.session_state.parse_queue = []


