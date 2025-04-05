import os
from dotenv import load_dotenv


def load_api_key() -> None:
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key

def get_available_documents() -> list:
    return [sub_folder.path.replace("./faiss-", "") for sub_folder in os.scandir() if sub_folder.is_dir() and "faiss-" in sub_folder.path]

