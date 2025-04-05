from pdf_processor import PDFProcessor
from text_chunker import TextChunker
from vector_store import VectorStoreManager
from qa_model import QuestionAnswering


class DocumentQAApp:
    def __init__(self, uploaded_pdf=None, folder_name=None, selected_language="Portuguese"):
        self.language = selected_language
        self.chunker = TextChunker()
        self.qa_model = QuestionAnswering()
        if uploaded_pdf:
            self.pdf_processor = PDFProcessor(uploaded_pdf)
            self.folder_name = uploaded_pdf.name
        elif folder_name:
            self.pdf_processor = None
            self.folder_name = folder_name
        else:
            raise ValueError("Either 'uploaded_pdf' or 'folder_name' must be provided.")

        self.vector_manager = VectorStoreManager(self.folder_name)

    def process_and_store(self):
        if not self.pdf_processor:
            raise RuntimeError("No uploaded PDF to process.")
        text = self.pdf_processor.extract_text()
        chunks = self.chunker.chunk_text(text)
        self.vector_manager.create_store(chunks)

    def answer_question(self, question: str):
        db = self.vector_manager.load_store()
        context_docs = db.similarity_search(question)
        response = self.qa_model.answer(question, context_docs, self.language)
        return response.content

