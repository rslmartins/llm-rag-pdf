from PyPDF2 import PdfReader
from streamlit.runtime.uploaded_file_manager import UploadedFile

class PDFProcessor:
    def __init__(self, pdf: UploadedFile):
        self.pdf = pdf

    def extract_text(self) -> str:
        text = ""
        pdf_reader = PdfReader(self.pdf)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
        return text

