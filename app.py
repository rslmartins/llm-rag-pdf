import streamlit as st
from utils import *


def main():
    st.set_page_config("LLM+RAG")
    st.header("Knowledge management through LLM+RAG")
    load_api_key()

    with st.sidebar:
        st.title("Menu:")
        selected_language = st.selectbox("Select a language:", ["Portuguese", "English"])
        selected_document = st.selectbox("Select a document:", get_available_documents())
        pdf = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button",
                               accept_multiple_files=False)
        if st.button("Process PDF"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks, pdf.name)
                st.rerun()
    user_question = st.text_input("Ask anything")

    if (st.button("Submit")):
        st.write(question_response(user_question, f"faiss-{selected_document}",
        selected_language) if user_question else "Please write a question!")

if __name__ == "__main__":
    main()

