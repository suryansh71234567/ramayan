import os
import sys
from dotenv import load_dotenv

# --- Add the parent directory to the Python path to import the original script's functions ---
# This is a flexible way to import from a separate folder.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

# Now, we can import the necessary components from the main pipeline script.
try:
    from scripts.sunderkand_pipeline import get_sunderkand_data, create_rag_vector_store, start_rag_chat
    from langchain_community.embeddings import HuggingFaceBgeEmbeddings
    from langchain_community.vectorstores import FAISS
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure your 'testing' folder is a sibling of the 'scripts' folder and you have the necessary libraries installed.")
    sys.exit(1)


# --- 1. Configuration for Testing ---
# These paths and models should match the original script
KANDA_NAME = "Sundara Kanda"
VECTOR_STORE_PATH = "sunderkand_faiss_index"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
OLLAMA_MODEL = "llama3:8b"
OLLAMA_API_URL = "http://localhost:11434"


def main():
    """
    Main function to load the vector store and start the chat interface.
    """
    load_dotenv()

    print("\n--- RAG Chatbot Test Script ---")

    # --- 2. Check for the vector store before attempting to load ---
    if not os.path.exists(VECTOR_STORE_PATH):
        print(f"Error: The vector store folder '{VECTOR_STORE_PATH}' was not found.")
        print("Please run 'sunderkand_pipeline.py' in the parent directory first to create the index.")
        sys.exit(1)

    # --- 3. Load the pre-existing vector store ---
    try:
        print(f"Loading vector store from '{VECTOR_STORE_PATH}'...")
        embeddings = HuggingFaceBgeEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
        print("Vector store loaded successfully.")
    except Exception as e:
        print(f"Failed to load vector store: {e}")
        print("Please ensure the embedding model is correctly specified and the index is not corrupted.")
        sys.exit(1)

    # --- 4. Start the interactive chat session with the loaded vector store ---
    start_rag_chat(vector_store, OLLAMA_MODEL, OLLAMA_API_URL)


if __name__ == "__main__":
    main()