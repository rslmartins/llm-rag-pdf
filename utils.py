import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


def get_pdf_text(pdf: UploadedFile) -> str:
    """
    Extracts and concatenates text from multiple PDF documents.

    Parameters:
    pdf_docs (List[str]): A list of file paths to the PDF documents.

    Returns:
    str: The extracted text from all pages of the provided PDFs.
    """
    text = ""
    # for pdf in pdf_docs:
    pdf_reader = PdfReader(pdf)  # Initialize PDF reader for each document
    for page in pdf_reader.pages:  # Iterate through all pages
        extracted_text = page.extract_text()
        if extracted_text:  # Check if text extraction was successful
            text += extracted_text
    return text


def get_text_chunks(text: str) -> List[str]:
    """
    Splits a large text into smaller overlapping chunks.

    Args:
        text (str): The full extracted text that needs to be chunked.

    Returns:
        List[str]: A list of text chunks.

    Explanation:
    - Uses `RecursiveCharacterTextSplitter` to divide large text into chunks.
    - `chunk_size=10000`: Each chunk will have a maximum of 10,000 characters.
    - `chunk_overlap=1000`: Ensures a 1,000-character overlap between consecutive chunks.
    - This overlap helps maintain context when processing the text for embeddings or NLP tasks.
    - Returns a list of split text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks: List[str], folder_name:str) -> None:
    """
    Creates and saves a FAISS vector store from a list of text chunks.

    This function generates dense vector representations (embeddings) 
    for the given text chunks using GoogleGenerativeAIEmbeddings and 
    stores them in a FAISS (Facebook AI Similarity Search) index.

    Args:
        text_chunks (List[str]): A list of text chunks to be converted into dense vectors.
    """
    # Generate dense vector embeddings for text chunks
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Create a FAISS vector store from the embeddings
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    
    # Save the FAISS index locally for future retrieval
    vector_store.save_local(f'faiss-{folder_name.replace(".pdf", "")}')



def retrieval_qa_chain():
    """
    Creates and returns a conversational chain for answering questions in Portuguese using Google's Gemini model.

    The chain takes a context and a question, then generates a detailed answer based on the context.
    If the context does not contain the answer, it responds with "não há informações disponíveis."

    Returns:
        chain: A LangChain question-answering chain configured to use Google's Gemini AI model.
    """
    prompt_template = """
    Answer the question, in {selected_language}, as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    # Prompt with Multiple prompt variable
    prompt_template = ChatPromptTemplate.from_template(prompt_template)
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                                   temperature=0.3)
    chain = prompt_template | model
    return chain


def question_response(user_question, folder_path="faiss_index", selected_language="Portuguese"):
    """
    Processes a user's question by searching for relevant documents using FAISS 
    and then generating a conversational response with Google's Gemini AI.

    Steps:
    1. Loads precomputed embeddings using Google's embedding model.
    2. Loads a FAISS vector store and performs a similarity search to find relevant documents.
    3. Uses the conversational chain to generate a response based on the retrieved documents.
    4. Prints and displays the response in Streamlit.

    Parameters:
        user_question (str): The user's question.

    Returns:
        None (prints and displays the AI-generated response)
    """
    # Load Google's embedding model for text vectorization.
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Load the FAISS vector store with precomputed embeddings,`allow_dangerous_deserialization` is for pickle files.
    new_db = FAISS.load_local(folder_path, embeddings, allow_dangerous_deserialization=True)

    # Perform a similarity search in the FAISS database to find relevant snippets.
    docs = new_db.similarity_search(user_question)

    # Get the conversational AI chain (uses Gemini model with a predefined prompt).
    chain = retrieval_qa_chain()

    # Generate a response using the retrieved documents as context.
    response = chain.invoke({"context": docs, "question": user_question, "selected_language": selected_language}, return_exceptions=True)
    print(response)
    return response.content

def get_available_documents() -> list:
    return [sub_folder.path.replace("./faiss-", "") for sub_folder in os.scandir() if sub_folder.is_dir() and "faiss-" in sub_folder.path]

def load_api_key() -> None:
    try:
        load_dotenv()
        os.getenv("GOOGLE_API_KEY")
    except:
        st.error("Please chek environment file")

