import os
import streamlit as st
from app_core import DocumentQAApp
from settings import load_api_key, get_available_documents
from vector_store import VectorStoreManager


def main():
    load_api_key()
    st.set_page_config("LLM+RAG")
    st.header("Knowledge management through LLM+RAG")

    with st.sidebar:
        st.title("Menu:")
        selected_language = st.selectbox("Select a language:", ["Portuguese", "English"])
        selected_document = st.selectbox("Select a document:", get_available_documents())
        uploaded_pdf = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button",
                                        accept_multiple_files=False)

        if uploaded_pdf and st.button("Process PDF"):
            with st.spinner("Processing..."):
                app = DocumentQAApp(uploaded_pdf, selected_language)
                app.process_and_store()
                st.success("PDF processed and stored!")
                st.rerun()

    user_question = st.text_input("Ask anything")

    if st.button("Submit"):
        if user_question and selected_document:
            app = DocumentQAApp(folder_name=selected_document, selected_language=selected_language)
            response = app.answer_question(user_question)
            st.write(response)
        else:
            st.warning("Please write a question and make sure a document is selected.")

if __name__ == "__main__":
    main()

