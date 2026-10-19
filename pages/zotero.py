# ============================================================
# 1. IMPORTS
# ============================================================

import json
import csv
import io
import re
from datetime import datetime

import pandas as pd
import streamlit as st

from pyzotero import Zotero

# ============================================================
# 2. STREAMLIT PAGE CONFIGURATION
# ============================================================



st.title("📚 Zotero Research Library")

st.write(
    "Search, explore, create, edit, and export references "
    "from your Zotero library."
)


# ============================================================
# 3. CREATE A ZOTERO CLIENT
# ============================================================

def create_zotero_client(library_id, library_type, api_key):
    """
    Create a PyZotero client.

    Parameters
    ----------
    library_id:
        Numeric user ID or group ID, represented as a string.

    library_type:
        "user" for a personal library, or "group" for a group library.

    api_key:
        Zotero Web API key.

    Returns
    -------
    Zotero:
        A PyZotero client bound to the specified library.
    """

    # Validate the library type before making an API request.
    if library_type not in ("user", "group"):
        raise ValueError(
            "Library type must be 'user' or 'group'."
        )

    # The client is permanently associated with this library.
    return Zotero(
        library_id,
        library_type,
        api_key,
    )


# ============================================================
# 4. LOAD CREDENTIALS
# ============================================================

# Read credentials from Streamlit secrets.
#
# This configuration is suitable for a single-user deployment
# or a trusted internal application.
#
# For an application serving multiple users, do not use one
# shared credential set for every user without considering
# authorization and data-isolation requirements.

try:
    zotero_config = st.secrets["zotero"]

    DEFAULT_LIBRARY_ID = str(
        zotero_config["library_id"]
    )

    DEFAULT_LIBRARY_TYPE = str(
        zotero_config["library_type"]
    )

    DEFAULT_API_KEY = str(
        zotero_config["api_key"]
    )

except (KeyError, FileNotFoundError):
    st.error(
        "Zotero credentials are missing. "
        "Configure .streamlit/secrets.toml first."
    )
    st.stop()


# ============================================================
# 5. SESSION STATE
# ============================================================

# Streamlit reruns the script after most user interactions.
# Session state lets us preserve choices between reruns.

if "selected_item_key" not in st.session_state:
    st.session_state.selected_item_key = None

if "confirm_delete_key" not in st.session_state:
    st.session_state.confirm_delete_key = None

if "library_cache_version" not in st.session_state:
    st.session_state.library_cache_version = 0


# ============================================================
# 6. SIDEBAR: CONNECTION SETTINGS
# ============================================================

st.sidebar.header("Connection")

# These fields initially contain values from secrets.toml.
# The API key field is masked by Streamlit.

library_id = st.sidebar.text_input(
    "Library ID",
    value=DEFAULT_LIBRARY_ID,
)

library_type = st.sidebar.selectbox(
    "Library type",
    options=["user", "group"],
    index=(
        0 if DEFAULT_LIBRARY_TYPE == "user" else 1
    ),
)

api_key = st.sidebar.text_input(
    "Zotero API key",
    value=DEFAULT_API_KEY,
    type="password",
)

# Create the client.
#
# In a larger application, you may want to cache clients
# separately for different users and credentials.
try:
    zot = create_zotero_client(
        library_id.strip(),
        library_type,
        api_key.strip(),
    )
except Exception as exc:
    st.error(f"Could not configure Zotero: {exc}")
    st.stop()


# ============================================================
# 7. HELPER FUNCTIONS FOR METADATA
# ============================================================

def get_item_data(item):
    """
    Extract the Zotero 'data' dictionary from an API response.

    Zotero item responses usually contain metadata under:
        item["data"]

    Some operations can return a different structure, so
    checking the type makes the helper more defensive.
    """

    if not isinstance(item, dict):
        return {}

    data = item.get("data", {})

    if not isinstance(data, dict):
        return {}

    return data


def get_creators_text(data):
    """
    Convert Zotero's structured creator list into readable text.

    Zotero stores creators as dictionaries rather than one
    plain author string.

    Example:
        [
            {
                "creatorType": "author",
                "firstName": "Ada",
                "lastName": "Lovelace"
            }
        ]
    """

    names = []

    for creator in data.get("creators", []):
        # Some creator types have a single-field name.
        name = creator.get("name")

        if not name:
            first = creator.get("firstName", "")
            last = creator.get("lastName", "")

            name = f"{first} {last}".strip()

        if name:
            names.append(name)

    return "; ".join(names)


def get_tags_text(data):
    """
    Convert Zotero's tag objects into a comma-separated string.
    """

    return ", ".join(
        tag.get("tag", "")
        for tag in data.get("tags", [])
        if tag.get("tag")
    )


def item_to_row(item):
    """
    Convert a Zotero item into a flat dictionary suitable
    for a pandas DataFrame.

    The original API response contains nested objects.
    A flat representation is easier to search, filter,
    display, and export.
    """

    data = get_item_data(item)

    return {
        "key": data.get("key", ""),
        "itemType": data.get("itemType", ""),
        "title": data.get("title", ""),
        "creators": get_creators_text(data),
        "date": data.get("date", ""),
        "publicationTitle": data.get(
            "publicationTitle", ""
        ),
        "publisher": data.get("publisher", ""),
        "DOI": data.get("DOI", ""),
        "ISBN": data.get("ISBN", ""),
        "url": data.get("url", ""),
        "abstractNote": data.get("abstractNote", ""),
        "tags": get_tags_text(data),
        "libraryCatalog": data.get(
            "libraryCatalog", ""
        ),
    }


def safe_text(value):
    """
    Convert a possibly missing value to a displayable string.
    """

    if value is None:
        return ""

    return str(value)


# ============================================================
# 8. CACHED READ OPERATIONS
# ============================================================

# Zotero API calls require network access and can be slower
# than local Python operations.
#
# Streamlit's cache reduces repeated calls during reruns.
#
# The cache key includes the credentials and library settings.
# This is convenient for a single-user app, but shared
# multi-user deployments need careful cache isolation.
#
# Avoid caching write operations.

@st.cache_data(ttl=60, show_spinner=False)
def load_top_items(
    library_id,
    library_type,
    api_key,
    cache_version,
    limit=100,
):
    """
    Load a bounded number of top-level references.

    'top' excludes child items such as notes and attachments.
    This is useful for a reference browser.
    """

    client = create_zotero_client(
        library_id,
        library_type,
        api_key,
    )

    return client.top(limit=limit)


@st.cache_data(ttl=60, show_spinner=False)
def load_collections(
    library_id,
    library_type,
    api_key,
    cache_version,
):
    """
    Retrieve collection metadata.

    Collection metadata includes each collection's key and name.
    """

    client = create_zotero_client(
        library_id,
        library_type,
        api_key,
    )

    return client.collections()


@st.cache_data(ttl=60, show_spinner=False)
def load_tags(
    library_id,
    library_type,
    api_key,
    cache_version,
):
    """
    Retrieve tags known to the library.
    """

    client = create_zotero_client(
        library_id,
        library_type,
        api_key,
    )

    return client.tags()


@st.cache_data(ttl=60, show_spinner=False)
def load_item_details(
    library_id,
    library_type,
    api_key,
    item_key,
    cache_version,
):
    """
    Retrieve one reference by its Zotero item key.
    """

    client = create_zotero_client(
        library_id,
        library_type,
        api_key,
    )

    return client.item(item_key)


@st.cache_data(ttl=60, show_spinner=False)
def load_child_items(
    library_id,
    library_type,
    api_key,
    item_key,
    cache_version,
):
    """
    Retrieve child items belonging to a reference.

    Child items may include attachments and notes.
    """

    client = create_zotero_client(
        library_id,
        library_type,
        api_key,
    )

    return client.children(item_key)


# ============================================================
# 9. CONNECTION TEST
# ============================================================

with st.sidebar:
    if st.button("Test connection", width="stretch"):
        try:
            # Fetch one item to verify access.
            sample = zot.top(limit=1)

            st.success(
                "Connection successful. "
                f"Sample size: {len(sample)}"
            )

        except Exception as exc:
            st.error(
                "Connection failed. Check your library ID, "
                f"API key, network, and permissions. Details: {exc}"
            )


# ============================================================
# 10. LOAD LIBRARY DATA
# ============================================================

# Load the top-level references, collections, and tags.
# Errors are caught so the application can explain what failed.

try:
    with st.spinner("Loading your Zotero library..."):
        raw_items = load_top_items(
            library_id,
            library_type,
            api_key,
            st.session_state.library_cache_version,
            limit=100,
        )

        raw_collections = load_collections(
            library_id,
            library_type,
            api_key,
            st.session_state.library_cache_version,
        )

        raw_tags = load_tags(
            library_id,
            library_type,
            api_key,
            st.session_state.library_cache_version,
        )

except Exception as exc:
    st.error(
        "Unable to retrieve library data. "
        "Check your Zotero credentials and internet connection."
    )

    # Useful for development, but avoid exposing detailed
    # exception messages in a public production application.
    with st.expander("Technical details"):
        st.code(str(exc))

    st.stop()


# Convert API results into flat tables.
rows = [item_to_row(item) for item in raw_items]

df = pd.DataFrame(rows)

if df.empty:
    # Define expected columns even when the library is empty.
    df = pd.DataFrame(
        columns=[
            "key",
            "itemType",
            "title",
            "creators",
            "date",
            "publicationTitle",
            "publisher",
            "DOI",
            "ISBN",
            "url",
            "abstractNote",
            "tags",
            "libraryCatalog",
        ]
    )


# ============================================================
# 11. LIBRARY OVERVIEW
# ============================================================

st.subheader("Library overview")

# These numbers describe the items loaded into this application.
# Since this demo loads at most 100 references, the item count
# below is NOT necessarily the total size of the library.

col1, col2, col3 = st.columns(3)

col1.metric(
    "References loaded",
    len(df),
)

col2.metric(
    "Collections",
    len(raw_collections),
)

col3.metric(
    "Distinct tags",
    len(raw_tags),
)

st.caption(
    "The reference table loads at most 100 top-level items. "
    "Increase the limit or use pagination for larger libraries."
)


# ============================================================
# 12. SEARCH AND FILTERS
# ============================================================

st.subheader("Search your references")

# A simple search box for the currently loaded references.
query = st.text_input(
    "Search title, author, abstract, DOI, or tag",
    placeholder="e.g. machine learning, Smith, 10.1234...",
)

# Filters appear in columns to save screen space.
filter_col1, filter_col2, filter_col3 = st.columns(3)

# Filter by item type.
available_types = sorted(
    value
    for value in df["itemType"].dropna().unique()
    if value
)

with filter_col1:
    selected_types = st.multiselect(
        "Item type",
        options=available_types,
        default=[],
    )

# Filter by publication year.
years = []

for value in df["date"].dropna():
    # Extract a four-digit year when one is present.
    match = re.search(r"\b\d{4}\b", str(value))

    if match:
        years.append(int(match.group()))

available_years = sorted(set(years))

with filter_col2:
    selected_years = st.multiselect(
        "Publication year",
        options=available_years,
        default=[],
    )

# Filter by tag.
all_tag_names = sorted(
    tag.get("tag", "")
    for tag in raw_tags
    if tag.get("tag")
)

with filter_col3:
    selected_tags = st.multiselect(
        "Tag",
        options=all_tag_names,
        default=[],
    )


# Apply filters to the DataFrame.
filtered_df = df.copy()

if query.strip():
    # Combine searchable fields into one string per row.
    searchable_columns = [
        "title",
        "creators",
        "abstractNote",
        "DOI",
        "tags",
        "publicationTitle",
    ]

    search_text = (
        filtered_df[searchable_columns]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
    )

    filtered_df = filtered_df[
        search_text.str.contains(
            query.strip(),
            case=False,
            regex=False,
        )
    ]

if selected_types:
    filtered_df = filtered_df[
        filtered_df["itemType"].isin(selected_types)
    ]

if selected_years:
    year_pattern = r"\b(" + "|".join(
        str(year) for year in selected_years
    ) + r")\b"

    filtered_df = filtered_df[
        filtered_df["date"]
        .fillna("")
        .astype(str)
        .str.contains(
            year_pattern,
            regex=True,
        )
    ]

if selected_tags:
    # Require every selected tag to appear in the reference.
    for tag in selected_tags:
        filtered_df = filtered_df[
            filtered_df["tags"]
            .fillna("")
            .apply(
                lambda text: tag in [
                    value.strip()
                    for value in text.split(",")
                ]
            )
        ]


# ============================================================
# 13. DISPLAY THE REFERENCE TABLE
# ============================================================

st.subheader("References")

st.write(
    f"Showing {len(filtered_df)} matching references "
    f"from {len(df)} loaded items."
)

# The user can sort and explore the visible metadata.
st.dataframe(
    filtered_df[
        [
            "key",
            "itemType",
            "title",
            "creators",
            "date",
            "publicationTitle",
            "DOI",
            "tags",
        ]
    ],
    width="stretch",
    hide_index=True,
)

# Select a reference to inspect in the next section.
if not filtered_df.empty:
    item_options = filtered_df["key"].tolist()

    selected_key = st.selectbox(
        "Choose a reference for detailed inspection",
        options=item_options,
        format_func=lambda key: (
            filtered_df.loc[
                filtered_df["key"] == key,
                "title",
            ].iloc[0]
            or key
        ),
    )

    st.session_state.selected_item_key = selected_key

else:
    st.info("No references match your filters.")


# ============================================================
# 14. INSPECT REFERENCE DETAILS
# ============================================================

st.subheader("Reference details")

selected_key = st.session_state.selected_item_key

if selected_key:
    try:
        # Retrieve the complete item rather than relying on
        # the smaller table representation.
        detailed_item = load_item_details(
            library_id,
            library_type,
            api_key,
            selected_key,
            st.session_state.library_cache_version,
        )

        detailed_data = get_item_data(detailed_item)

        # Display selected fields.
        st.markdown(
            f"### {detailed_data.get('title', 'Untitled')}"
        )

        st.write(
            "**Type:**",
            detailed_data.get("itemType", ""),
        )

        st.write(
            "**Creators:**",
            get_creators_text(detailed_data),
        )

        st.write(
            "**Date:**",
            detailed_data.get("date", ""),
        )

        st.write(
            "**Publication:**",
            detailed_data.get("publicationTitle", ""),
        )

        st.write(
            "**Publisher:**",
            detailed_data.get("publisher", ""),
        )

        st.write(
            "**DOI:**",
            detailed_data.get("DOI", ""),
        )

        st.write(
            "**URL:**",
            detailed_data.get("url", ""),
        )

        abstract = detailed_data.get("abstractNote", "")

        with st.expander("Abstract"):
            st.write(abstract or "No abstract available.")

        with st.expander("Complete Zotero metadata"):
            # This exposes all fields returned by the API.
            st.json(detailed_data)

        # ----------------------------------------------------
        # CHILD ITEMS
        # ----------------------------------------------------

        with st.expander("Notes and attachments"):
            child_items = load_child_items(
                library_id,
                library_type,
                api_key,
                selected_key,
                st.session_state.library_cache_version,
            )

            if child_items:
                for child in child_items:
                    child_data = get_item_data(child)

                    st.write(
                        f"**{child_data.get('itemType', '')}:** "
                        f"{child_data.get('title', 'Untitled')}"
                    )

                    # Display note contents if returned.
                    if child_data.get("note"):
                        st.markdown(
                            child_data["note"],
                            unsafe_allow_html=False,
                        )

                    # Show attachment metadata without assuming
                    # that the file is directly downloadable.
                    if child_data.get("itemType") == "attachment":
                        st.json(
                            {
                                "key": child_data.get("key"),
                                "title": child_data.get("title"),
                                "contentType": child_data.get(
                                    "contentType"
                                ),
                                "linkMode": child_data.get(
                                    "linkMode"
                                ),
                                "url": child_data.get("url"),
                            }
                        )

            else:
                st.info("No child items found.")

    except Exception as exc:
        st.error(f"Unable to load reference details: {exc}")


# ============================================================
# 15. BROWSE COLLECTIONS
# ============================================================

st.subheader("Collections")

# Display the collections returned by Zotero.
collection_rows = []

for collection in raw_collections:
    data = get_item_data(collection)

    # Collection objects may expose their data differently
    # from ordinary item objects. Check the API response.
    if not data:
        data = collection.get("data", collection)

    collection_rows.append(
        {
            "key": data.get("key", ""),
            "name": data.get("name", ""),
            "parentCollection": data.get(
                "parentCollection", ""
            ),
        }
    )

collections_df = pd.DataFrame(collection_rows)

if not collections_df.empty:
    st.dataframe(
        collections_df,
        width="stretch",
        hide_index=True,
    )

    # Let the user inspect a collection's contents.
    collection_options = {
        row["name"]: row["key"]
        for row in collection_rows
        if row["name"] and row["key"]
    }

    if collection_options:
        selected_collection_name = st.selectbox(
            "Browse a collection",
            options=list(collection_options.keys()),
        )

        selected_collection_key = collection_options[
            selected_collection_name
        ]

        if st.button("Load collection references"):
            try:
                collection_items = zot.collection_items(
                    selected_collection_key
                )

                collection_rows_for_table = [
                    item_to_row(item)
                    for item in collection_items
                ]

                st.dataframe(
                    pd.DataFrame(
                        collection_rows_for_table
                    ),
                    width="stretch",
                    hide_index=True,
                )

            except Exception as exc:
                st.error(
                    f"Could not load collection: {exc}"
                )

else:
    st.info("No collections were found.")


# ============================================================
# 16. BROWSE TAGS
# ============================================================

st.subheader("Tags")

if raw_tags:
    # Zotero tag objects can contain counts and other metadata.
    # Extract the tag name for a simple display.
    tag_names = sorted(
        tag.get("tag", "")
        for tag in raw_tags
        if tag.get("tag")
    )

    st.write(", ".join(tag_names))

else:
    st.info("No tags were found.")


# ============================================================
# 17. CREATE A NEW REFERENCE
# ============================================================

st.subheader("Create a reference")

st.write(
    "Add a journal article or a book to your Zotero library."
)

with st.form("create_reference_form"):

    # Zotero item types determine which metadata fields
    # are meaningful for the reference.
    new_item_type = st.selectbox(
        "Reference type",
        options=["journalArticle", "book"],
    )

    new_title = st.text_input("Title")

    new_author_first = st.text_input(
        "First author's first name"
    )

    new_author_last = st.text_input(
        "First author's last name"
    )

    new_date = st.text_input(
        "Publication date",
        placeholder="2025",
    )

    new_doi = st.text_input("DOI")

    new_publication = st.text_input(
        "Journal title (for journal articles)"
    )

    new_publisher = st.text_input(
        "Publisher (for books)"
    )

    new_abstract = st.text_area("Abstract")

    new_tags = st.text_input(
        "Tags, separated by commas"
    )

    create_submitted = st.form_submit_button(
        "Create reference"
    )

if create_submitted:

    # A title is the minimum useful field for this demo.
    if not new_title.strip():
        st.error("Please enter a title.")

    else:
        # Construct a Zotero item-data object.
        #
        # Field names must match Zotero's schema.
        # Different item types support different fields.
        new_item = {
            "itemType": new_item_type,
            "title": new_title.strip(),
            "date": new_date.strip(),
            "abstractNote": new_abstract.strip(),
            "creators": [],
            "tags": [],
        }

        # Add the first author if a name was provided.
        if new_author_first.strip() or new_author_last.strip():
            new_item["creators"].append(
                {
                    "creatorType": "author",
                    "firstName": new_author_first.strip(),
                    "lastName": new_author_last.strip(),
                }
            )

        # Add the DOI when provided.
        if new_doi.strip():
            new_item["DOI"] = new_doi.strip()

        # Only populate the field appropriate to the item type.
        if (
            new_item_type == "journalArticle"
            and new_publication.strip()
        ):
            new_item["publicationTitle"] = (
                new_publication.strip()
            )

        if (
            new_item_type == "book"
            and new_publisher.strip()
        ):
            new_item["publisher"] = new_publisher.strip()

        # Zotero expects tags as objects, not one string.
        new_item["tags"] = [
            {"tag": tag.strip()}
            for tag in new_tags.split(",")
            if tag.strip()
        ]

        try:
            # Write the item to Zotero.
            #
            # Creating a reference changes the library, so this
            # operation should not be wrapped in a cached function.
            result = zot.create_items([new_item])

            st.success(
                "Creation request completed. "
                "Check the result below."
            )

            st.json(result)

            # Invalidate cached data after a successful write.
            st.session_state.library_cache_version += 1
            st.cache_data.clear()

        except Exception as exc:
            st.error(
                f"Could not create the reference: {exc}"
            )


# ============================================================
# 18. EDIT AN EXISTING REFERENCE
# ============================================================

st.subheader("Edit a reference")

if selected_key:

    # Reuse the selected reference's metadata.
    try:
        editable_item = load_item_details(
            library_id,
            library_type,
            api_key,
            selected_key,
            st.session_state.library_cache_version,
        )

        editable_data = get_item_data(editable_item)

        with st.form("edit_reference_form"):

            edited_title = st.text_input(
                "Title",
                value=editable_data.get("title", ""),
            )

            edited_date = st.text_input(
                "Date",
                value=editable_data.get("date", ""),
            )

            edited_doi = st.text_input(
                "DOI",
                value=editable_data.get("DOI", ""),
            )

            edited_abstract = st.text_area(
                "Abstract",
                value=editable_data.get("abstractNote", ""),
            )

            edit_submitted = st.form_submit_button(
                "Save changes"
            )

        if edit_submitted:

            # Start from the complete original item response.
            #
            # Updating only a small custom dictionary can omit
            # metadata or version information needed by the API.
            updated_item = dict(editable_item)
            updated_data = dict(editable_data)

            # Change only the fields this form edits.
            updated_data["title"] = edited_title.strip()
            updated_data["date"] = edited_date.strip()
            updated_data["DOI"] = edited_doi.strip()
            updated_data["abstractNote"] = (
                edited_abstract.strip()
            )

            updated_item["data"] = updated_data

            try:
                result = zot.update_item(updated_item)

                st.success(
                    f"Update completed: {result}"
                )

                # Refresh cached references.
                st.session_state.library_cache_version += 1
                st.cache_data.clear()

            except Exception as exc:
                st.error(
                    f"Could not update the reference: {exc}"
                )

    except Exception as exc:
        st.error(f"Unable to prepare edit form: {exc}")

else:
    st.info("Select a reference above to edit it.")


# ============================================================
# 19. DELETE A REFERENCE
# ============================================================

st.subheader("Delete a reference")

st.warning(
    "Deletion changes your Zotero library. "
    "Use this operation carefully."
)

if selected_key:

    # Use a two-step confirmation to reduce accidental deletion.
    if st.button("Prepare deletion"):
        st.session_state.confirm_delete_key = selected_key

    if (
        st.session_state.confirm_delete_key
        == selected_key
    ):
        st.error(
            "Confirm that you want to delete this reference."
        )

        confirm = st.checkbox(
            "I understand that this will delete the reference."
        )

        if st.button("Confirm deletion"):

            if not confirm:
                st.error(
                    "Check the confirmation box first."
                )

            else:
                try:
                    # This is a destructive API operation.
                    result = zot.delete_item(selected_key)

                    st.success(
                        f"Deletion completed: {result}"
                    )

                    st.session_state.confirm_delete_key = None
                    st.session_state.selected_item_key = None

                    st.session_state.library_cache_version += 1
                    st.cache_data.clear()

                except Exception as exc:
                    st.error(
                        f"Could not delete the reference: {exc}"
                    )


# ============================================================
# 20. EXPORT REFERENCES AS CSV
# ============================================================

st.subheader("Export references")

# Export the currently filtered references.
#
# CSV is convenient for spreadsheets and data analysis.
csv_data = filtered_df.to_csv(
    index=False
).encode("utf-8-sig")

st.download_button(
    label="Download filtered references as CSV",
    data=csv_data,
    file_name="zotero_references.csv",
    mime="text/csv",
)


# ============================================================
# 21. EXPORT REFERENCES AS JSON
# ============================================================

# JSON preserves structured metadata more naturally than CSV.
#
# This export contains the flattened table representation,
# not every field from the original Zotero API responses.

json_data = filtered_df.to_json(
    orient="records",
    indent=2,
    force_ascii=False,
)

st.download_button(
    label="Download filtered references as JSON",
    data=json_data,
    file_name="zotero_references.json",
    mime="application/json",
)


# ============================================================
# 22. EXPORT SELECTED REFERENCES AS BASIC BIBTEX
# ============================================================

def bibtex_escape(value):
    """
    Escape a few characters commonly found in BibTeX fields.

    This is a simple helper, not a complete BibTeX parser.
    """

    value = safe_text(value)

    return (
        value
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def row_to_bibtex(row):
    """
    Convert one flattened row to a basic BibTeX entry.

    This is a teaching example. For production-quality exports,
    prefer a dedicated bibliographic conversion library.
    """

    item_type = safe_text(row.get("itemType", ""))

    # Map a few Zotero item types to BibTeX entry types.
    type_mapping = {
        "journalArticle": "article",
        "book": "book",
        "bookSection": "incollection",
        "conferencePaper": "inproceedings",
        "thesis": "phdthesis",
        "report": "techreport",
    }

    bib_type = type_mapping.get(item_type, "misc")

    # Zotero item keys are suitable as stable identifiers
    # within the library, but can contain unexpected characters.
    raw_key = safe_text(row.get("key", "reference"))

    citation_key = re.sub(
        r"[^A-Za-z0-9_:-]",
        "_",
        raw_key,
    )

    fields = {
        "title": row.get("title", ""),
        "author": row.get("creators", "").replace("; ", " and "),
        "year": row.get("date", ""),
        "journal": row.get("publicationTitle", ""),
        "publisher": row.get("publisher", ""),
        "doi": row.get("DOI", ""),
        "url": row.get("url", ""),
    }

    # Omit empty fields.
    lines = [
        f"@{bib_type}{{{citation_key},"
    ]

    for field_name, field_value in fields.items():
        if safe_text(field_value).strip():
            lines.append(
                f"  {field_name} = "
                f"{{{bibtex_escape(field_value)}}},"
            )

    lines.append("}")

    return "\n".join(lines)


bibtex_output = "\n\n".join(
    row_to_bibtex(row)
    for _, row in filtered_df.iterrows()
)

st.download_button(
    label="Download basic BibTeX",
    data=bibtex_output,
    file_name="zotero_references.bib",
    mime="application/x-bibtex",
)


# ============================================================
# 23. EXPORT REFERENCES AS RIS
# ============================================================

def row_to_ris(row):
    """
    Convert a flattened reference into a basic RIS record.

    RIS uses tagged lines such as:
        TY  - JOUR
        TI  - Article title
        AU  - Author name
        ER  -

    A production exporter should map additional fields and
    item types according to the RIS specification.
    """

    item_type = safe_text(row.get("itemType", ""))

    type_mapping = {
        "journalArticle": "JOUR",
        "book": "BOOK",
        "bookSection": "CHAP",
        "conferencePaper": "CONF",
        "thesis": "THES",
        "report": "RPRT",
    }

    ris_type = type_mapping.get(item_type, "GEN")

    lines = [f"TY  - {ris_type}"]

    title = safe_text(row.get("title", ""))

    if title:
        lines.append(f"TI  - {title}")

    authors = safe_text(row.get("creators", ""))

    if authors:
        # The flattened author field uses semicolons.
        for author in authors.split("; "):
            lines.append(f"AU  - {author}")

    date = safe_text(row.get("date", ""))

    if date:
        lines.append(f"PY  - {date}")

    journal = safe_text(
        row.get("publicationTitle", "")
    )

    if journal:
        lines.append(f"JO  - {journal}")

    doi = safe_text(row.get("DOI", ""))

    if doi:
        lines.append(f"DO  - {doi}")

    url = safe_text(row.get("url", ""))

    if url:
        lines.append(f"UR  - {url}")

    lines.append("ER  -")

    return "\n".join(lines)


ris_output = "\n\n".join(
    row_to_ris(row)
    for _, row in filtered_df.iterrows()
)

st.download_button(
    label="Download basic RIS",
    data=ris_output,
    file_name="zotero_references.ris",
    mime="application/x-research-info-systems",
)


# ============================================================
# 24. BASIC RESEARCH STATISTICS
# ============================================================

st.subheader("Research statistics")

# Analyze the currently loaded reference sample.
# Remember that this is not necessarily the whole library.

if not df.empty:

    # Count references by Zotero item type.
    type_counts = (
        df["itemType"]
        .fillna("Unknown")
        .value_counts()
    )

    st.write("References by item type")

    st.bar_chart(type_counts)

    # Extract publication years for a simple trend chart.
    year_values = []

    for value in df["date"].fillna(""):
        match = re.search(
            r"\b\d{4}\b",
            str(value),
        )

        if match:
            year_values.append(int(match.group()))

    if year_values:
        year_counts = (
            pd.Series(year_values)
            .value_counts()
            .sort_index()
        )

        st.write("References by publication year")

        st.bar_chart(year_counts)

    # Count the most common author strings.
    #
    # Because the table stores authors in one flattened field,
    # this simple analysis does not distinguish author order
    # or normalize different spellings of the same person.
    author_counts = {}

    for authors in df["creators"].fillna(""):
        for author in str(authors).split(";"):
            author = author.strip()

            if author:
                author_counts[author] = (
                    author_counts.get(author, 0) + 1
                )

    if author_counts:
        top_authors = (
            pd.Series(author_counts)
            .sort_values(ascending=False)
            .head(10)
        )

        st.write("Most frequent author names")

        st.bar_chart(top_authors)


# ============================================================
# 25. REFRESH THE LIBRARY
# ============================================================

st.sidebar.divider()

if st.sidebar.button(
    "Refresh library",
    width="stretch",
):
    # Changing the cache version causes subsequent reads
    # to use a different cache key.
    st.session_state.library_cache_version += 1

    # Clear all cached data for this Streamlit process.
    st.cache_data.clear()

    # Rerun the app to fetch fresh data.
    st.rerun()


# ============================================================
# 26. FOOTER
# ============================================================

st.divider()

st.caption(
    "Zotero Research Library — built with Streamlit and PyZotero. "
    "Review API permissions, data privacy, and write operations "
    "before deploying this application."
)