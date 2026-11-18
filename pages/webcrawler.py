import os
import re
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

import streamlit as st
import pandas as pd
import functions


# Configuration: Set your target folder (e.g., MD_FOLDER from your RAG app)
if "folder_paths" in st.session_state and "MD_FOLDER" in st.session_state.folder_paths:
    MD_FOLDER = st.session_state.folder_paths["MD_FOLDER"]
else:
    MD_FOLDER = "./mds"
    functions.add_log(f"⚠️ MD_FOLDER not found in session_state. Defaulting to {MD_FOLDER}")

with st.sidebar:
    with st.expander("🌐 Web Documentation Scraper", expanded=True):
        # FIX: Changed default URL to the correct canonical root
        target_index_url = st.text_input(
            "Index URL", 
            value="https://tkdocs.com/tutorial/",
            help="The table of contents or index page to crawl."
        )
        
        subfolder_name = st.text_input(
            "Output Subfolder", 
            value="tkdocs_tutorial",
            help=f"Folder created inside {MD_FOLDER}."
        )
        
        crawl_timeout = st.number_input("Request Timeout (s)", 5, 60, 15)
        functions.add_log(f"📥 Fetching index from {target_index_url}...")


st.markdown("# 🌐 Web Documentation Scraper & Converter")
st.write(
    "Automatically crawl online documentation indexes (like TkDocs), convert HTML chapters "
    "into clean Markdown, prepend your `--- METADATA ---` headers, and save them into your "
    "Obsidian-ready Markdown folder for RAG indexing."
)

# Main action button
if st.button("🚀 Start Crawling and Converting Documentation", type="primary", use_container_width=True):
    successes = []
    failures = []
        
    output_subfolder = os.path.join(MD_FOLDER, subfolder_name)
    os.makedirs(output_subfolder, exist_ok=True)
    
    with st.spinner(f"📥 Connecting to {target_index_url}..."):
        try:
            resp = requests.get(target_index_url, timeout=crawl_timeout)
            if resp.status_code != 200:
                st.error(f"Failed to fetch index page. Status code: {resp.status_code}")
                st.stop()
                
            soup = BeautifulSoup(resp.text, 'html.parser')
            
            tutorial_links = set()
            # Ensure base domain handles trailing slashes correctly for relative resolution
            base_domain_parts = target_index_url if target_index_url.endswith('/') else target_index_url.rsplit('/', 1)[0] + '/'
            
            for a_tag in soup.find_all('a', href=True):
                href = str(a_tag['href'])
                if not href.startswith('http') and not href.startswith('#'):
                    clean_href = href.lstrip('./')
                    # FIX: More flexible check for chapter pages containing .html
                    if '.html' in clean_href and 'index' not in clean_href:
                        full_url = urljoin(base_domain_parts, clean_href)
                        tutorial_links.add(full_url)
            
            sorted_links = sorted(list(tutorial_links))
            total_pages = len(sorted_links)
            
            if total_pages == 0:
                st.warning("No valid `.html` links discovered on this index page.")
                st.stop()
                
            st.success(f"🔗 Discovered {total_pages} tutorial chapters to download!")

            progress_bar = st.progress(0, text="Initializing crawler...")
            success_count = 0
            
            for idx, page_url in enumerate(sorted_links, start=1):
                parsed_url = urlparse(page_url)
                filename = os.path.basename(parsed_url.path)
                md_filename = filename.replace('.html', '.md')
                
                progress_bar.progress(
                    idx / total_pages, 
                    text=f"Crawling [{idx}/{total_pages}]: `{filename}`"
                )
                
                try:
                    page_resp = requests.get(page_url, timeout=crawl_timeout)
                    if page_resp.status_code != 200:
                        failures.append({
                            "url": page_url, 
                            "reason": f"HTTP Status Error: {page_resp.status_code}"
                        })
                        continue
                        
                    page_soup = BeautifulSoup(page_resp.text, 'html.parser')
                    
                    content_div = (
                        page_soup.find('div', class_='content') or 
                        page_soup.find('article') or 
                        page_soup.find('main') or 
                        page_soup.find('body')
                    )
                    
                    if not content_div:
                        failures.append({
                            "url": page_url, 
                            "reason": "Could not find main content container in HTML"
                        })
                        continue

                    markdown_body = md(str(content_div), heading_style="ATX")
                    
                    # FIX: Unindented multi-line string to prevent YAML frontmatter corruption
                    frontmatter = f"""--- METADATA ---
source_file: "{subfolder_name}/{md_filename}"
DOI: "None"
URL: "{page_url}"
pages: 1
--- METADATA ---
"""
                    final_document = frontmatter + "\n" + markdown_body

                    save_path = os.path.join(output_subfolder, md_filename)
                    with open(save_path, "w", encoding="utf-8") as out_f:
                        out_f.write(final_document)
                        
                    successes.append(page_url)
                    
                except requests.exceptions.Timeout:
                    failures.append({"url": page_url, "reason": "Request Timed Out"})
                except Exception as inner_e:
                    failures.append({"url": page_url, "reason": str(inner_e)})
            
            progress_bar.empty()
            
        except Exception as e:
            st.error(f"❌ Crawler failed: {e}")

    total_attempted = len(sorted_links) if 'sorted_links' in locals() else 0
    success_count = len(successes)
    fail_count = len(failures)

    if total_attempted > 0:
        if fail_count == 0:
            st.success(f"✅ Perfect Run! Successfully converted all {success_count} chapters into `{output_subfolder}`.")
        elif success_count > 0:
            st.warning(f"⚠️ Partial Completion: {success_count} succeeded, but {fail_count} of {total_attempted} chapters failed.")
        else:
            st.error(f"❌ Complete Failure: All {total_attempted} chapters failed to crawl.")

    # --- INTERACTIVE ERROR ANALYSIS TABLE ---
    if failures:
        with st.expander(f"🔍 Detailed Error Analysis ({fail_count} failures)", expanded=True):
            st.write(
                "Review the error table below. You can click directly on any **Failed URL** "
                "to open it in your browser and inspect why it failed."
            )

            df_failures = pd.DataFrame(failures)
            df_failures.columns = ["Failed URL", "Error Reason"]

            # FIX: Removed invalid lambda expression from LinkColumn
            st.dataframe(
                df_failures,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Failed URL": st.column_config.LinkColumn(
                        "Failed URL (Clickable)",
                        help="Click to open the target page in your browser",
                        width="large",
                    ),
                    "Error Reason": st.column_config.TextColumn(
                        "Error Reason / HTTP Status",
                        width="large",
                    ),
                },
            )

    if success_count > 0:
        st.info("💡 **Next Step:** Go to your RAG Workbench page and click **'Save Config & Build / Update DB'** to index the newly converted files.")




# Define standard browser headers so the server doesn't block the request
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# Main action button
if st.button("🚀 Start Crawling and Converting Documentation - Version 2", type="primary", use_container_width=True):
    successes = []
    failures = []
        
    output_subfolder = os.path.join(MD_FOLDER, subfolder_name)
    os.makedirs(output_subfolder, exist_ok=True)
    
    with st.spinner(f"📥 Connecting to {target_index_url}..."):
        try:
            # PASS HEADERS HERE
            resp = requests.get(target_index_url, headers=HEADERS, timeout=crawl_timeout)
            if resp.status_code != 200:
                st.error(f"Failed to fetch index page. Status code: {resp.status_code}")
                st.stop()
                
            soup = BeautifulSoup(resp.text, 'html.parser')
            
            tutorial_links = set()
            base_domain_parts = target_index_url if target_index_url.endswith('/') else target_index_url.rsplit('/', 1)[0] + '/'
            
            for a_tag in soup.find_all('a', href=True):
                href = str(a_tag['href'])
                if not href.startswith('http') and not href.startswith('#'):
                    clean_href = href.lstrip('./')
                    if '.html' in clean_href and 'index' not in clean_href:
                        full_url = urljoin(base_domain_parts, clean_href)
                        tutorial_links.add(full_url)
            
            sorted_links = sorted(list(tutorial_links))
            total_pages = len(sorted_links)
            
            if total_pages == 0:
                st.warning("No valid `.html` links discovered on this index page. Check if the URL is correct or if headers are required.")
                st.stop()
                
            st.success(f"🔗 Discovered {total_pages} tutorial chapters to download!")
            
            progress_bar = st.progress(0, text="Initializing crawler...")
            success_count = 0
            
            for idx, page_url in enumerate(sorted_links, start=1):
                parsed_url = urlparse(page_url)
                filename = os.path.basename(parsed_url.path)
                md_filename = filename.replace('.html', '.md')
                
                progress_bar.progress(
                    idx / total_pages, 
                    text=f"Crawling [{idx}/{total_pages}]: `{filename}`"
                )
                
                try:
                    # PASS HEADERS HERE TOO
                    page_resp = requests.get(page_url, headers=HEADERS, timeout=crawl_timeout)
                    if page_resp.status_code != 200:
                        failures.append({
                            "url": page_url, 
                            "reason": f"HTTP Status Error: {page_resp.status_code}"
                        })
                        continue
                        
                    page_soup = BeautifulSoup(page_resp.text, 'html.parser')
                    
                    content_div = (
                        page_soup.find('div', class_='content') or 
                        page_soup.find('article') or 
                        page_soup.find('main') or 
                        page_soup.find('body')
                    )
                    
                    if not content_div:
                        failures.append({
                            "url": page_url, 
                            "reason": "Could not find main content container in HTML"
                        })
                        continue

                    markdown_body = md(str(content_div), heading_style="ATX")
                    
                    frontmatter = f"""
                        --- METADATA ---
                        source_file: "{subfolder_name}/{md_filename}"
                        DOI: "None"
                        URL: "{page_url}"
                        pages: 1
                        --- METADATA ---
                        """
                    final_document = frontmatter + "\n" + markdown_body

                    save_path = os.path.join(output_subfolder, md_filename)
                    with open(save_path, "w", encoding="utf-8") as out_f:
                        out_f.write(final_document)
                        
                    successes.append(page_url)
                    
                except requests.exceptions.Timeout:
                    failures.append({"url": page_url, "reason": "Request Timed Out"})
                except Exception as inner_e:
                    failures.append({"url": page_url, "reason": str(inner_e)})
            
            progress_bar.empty()
            
        except Exception as e:
            st.error(f"❌ Crawler failed: {e}")