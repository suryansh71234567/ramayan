import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS

from scripts.sunderkand_pipeline import start_rag_chat  # replace with actual import

def main():
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")  # store in .env

    VECTOR_STORE_PATH = "sunderkand_faiss_index"
    EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

    if not os.path.exists(VECTOR_STORE_PATH):
        print("Vector store not found. Please build it first.")
        return

    embeddings = HuggingFaceBgeEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)

    start_rag_chat(vector_store, api_key, model="mistralai/mistral-7b-instruct")

if __name__ == "__main__":
    main()
