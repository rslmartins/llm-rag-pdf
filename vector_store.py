from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

class VectorStoreManager:
    def __init__(self, folder_name: str):
        self.folder_name = folder_name.replace(".pdf", "")
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    def create_store(self, chunks):
        vector_store = FAISS.from_texts(chunks, embedding=self.embeddings)
        vector_store.save_local(f"faiss-{self.folder_name}")

    def load_store(self):
        return FAISS.load_local(f"faiss-{self.folder_name}", self.embeddings, allow_dangerous_deserialization=True)

