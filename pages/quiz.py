import streamlit as st    
from io import StringIO



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

            st.number_input("How many questions per document?", min_value=1, step=1, value=5)

            if st.button("Start QUIZ", disabled=st.session_state.is_parsing, type="primary", width="stretch"):
                st.session_state.is_quizzing = True


if st.session_state.is_quizzing:
    for filepath in uploaded_files:
        # To convert to a string based IO:
        stringio = StringIO(filepath.getvalue().decode("utf-8"))
        #st.write(stringio)

        # To read file as string:
        string_data = stringio.read()
        st.write(string_data)

        st.info(f"Questions regarding {filepath.name}:")

        


