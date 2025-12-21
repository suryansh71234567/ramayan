import json
import requests
import os
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
import torch


# Load environment variables (e.g., API keys, though not needed for Ollama)
load_dotenv()

# --- 1. Configuration: Modify these values for future updates ---
DATASET_URL = "https://raw.githubusercontent.com/AshuVj/Valmiki_Ramayan_Dataset/refs/heads/main/data/Valmiki_Ramayan_Shlokas.json"
DATA_FILE_PATH = os.path.join("data", "Valmiki_Ramayan_Shlokas.json")
KANDA_NAME = "Sundara Kanda"
VECTOR_STORE_PATH = "sunderkand_faiss_index"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"  # A powerful multilingual model for Sanskrit/English
OLLAMA_API_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3"

# --- 2. Data Ingestion & Preprocessing ---
def get_ramayana_data(url: str, file_path: str, kanda: str = None) -> list:
    """
    Downloads the dataset if it doesn't exist and filters for the specified kanda.
    
    Args:
        url (str): URL of the raw JSON dataset.
        file_path (str): Local path to save the dataset.
        kanda (str): The name of the kanda to filter.
        
    Returns:
        list: A list of shloka dictionaries for the specified kanda.
    """
    if not os.path.exists("data"):
        os.makedirs("data")

    if not os.path.exists(file_path):
        print("Dataset not found. Downloading...")
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an HTTPError for bad responses
            data = response.json()
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            print("Download complete.")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading data: {e}")
            return []
    else:
        print("Dataset already exists. Reading from file...")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    if kanda:
            filtered_data = [item for item in data if item.get("kanda") == kanda]
            print(f"Found {len(filtered_data)} shlokas in '{kanda}'.")
            return filtered_data
    else:
        print(f"No specific Kanda selected. Loading ALL {len(data)} shlokas from Ramayana.")
        return data

# --- 3. Indexing & Vector Store Creation ---
def create_rag_vector_store(data: list, store_path: str, model_name: str) -> FAISS:
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        print(f"✅ CUDA Detected. Running on GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps" # Apple Silicon Metal Performance Shaders
        print("✅ Apple MPS Detected. Running on Neural Engine/GPU.")
    else:
        print("⚠️ No GPU detected. Running on CPU (this will be slow for 24k docs).")

    # 2. Configure Embedding Model with Batching
    print(f"Initializing {model_name} on {device}...")
    model_kwargs = {'device': device}
    encode_kwargs = {
        'normalize_embeddings': True, 
        'batch_size': 64  # Process 64 docs at once. Lower this if you get OutOfMemory errors.
    }
    
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    # 3. Check for existing index
    if os.path.exists(store_path):
        print(f"Vector store found at '{store_path}'. Loading...")
        return FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)

    print("\n--- Starting Full Ramayana Indexing ---")

    print("\n--- Starting RAG Indexing Pipeline ---")
    documents = []
    for item in data:
        # Create a unique citation ID for each shloka
        citation_id = f"{item['kanda'].replace(' ', '')}_sarga_{item['sarga']}_shloka_{item['shloka']}"

        # Combine all relevant text for rich context
        content = (
            f"Original: {item.get('shloka_text', 'N/A')}\n"
            f"Explanation: {item.get('explanation', 'N/A')}\n"
            f"Comments: {item.get('comments', 'N/A')}"
        )
        
        # Create a Document object with content and citation metadata
        documents.append(
            Document(
                page_content=content, 
                metadata={"citation_id": citation_id}
            )
        )
    
    print(f"Created {len(documents)} documents for indexing.")
    print(f"Generating embeddings with {model_name}...")
    
    # Initialize the embedding model
    embeddings = HuggingFaceBgeEmbeddings(model_name=model_name)
    
    # Create the FAISS vector store
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    # Save the index to a folder
    vectorstore.save_local(store_path)
    print(f"FAISS vector store saved to '{store_path}'.")
    
    return vectorstore

# --- 4. RAG Chatbot with Citation ---
import requests
from langchain_core.prompts import ChatPromptTemplate


def start_rag_chat(vectorstore, api_key: str, model: str = "deepseek/deepseek-chat"):
    """
    Starts an interactive chatbot session using a remote API (OpenRouter example).
    """
    print("\n--- Starting Interactive Chat Session ---")
    print(f"Chatbot is powered by {model}. Ask me anything about Sundara Kanda!")
    print("Type 'exit' or 'quit' to end the session.")

    # Prompt template
    template = """
    You are an expert on the Valmiki Ramayana. 
    Answer the user's question based ONLY on the provided context.
    If the answer is not in the context, say you cannot answer.
    Cite verse references from metadata like [citation_id].

    Context:
    {context}

    Question: {question}
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Retrieval chain (we'll call API manually inside loop)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    while True:
        user_query = input("\n[You]: ")
        if user_query.lower() in ["exit", "quit"]:
            print("Ending session. Goodbye!")
            break

        # Get relevant docs
        docs = retriever.get_relevant_documents(user_query)
        context_text = "\n\n".join([d.page_content for d in docs])

        # Prepare API request
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": template.format(context=context_text, question=user_query)}
            ]
        }

        # Call API
        try:
            resp = requests.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            answer = resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            answer = f"Error calling API: {e}"

        print(f"\n[Chatbot]: {answer}")

        # Show sources
        if docs:
            print("\nSources:")
            for doc in docs:
                print(f"- {doc.metadata.get('citation_id', 'N/A')}")

'''def start_rag_chat1(vectorstore: FAISS, llm_model: str, api_url: str):
    """
    Starts an interactive chatbot session.
    
    Args:
        vectorstore (FAISS): The populated vector store.
        llm_model (str): Name of the LLM model (e.g., 'llama3').
        api_url (str): The URL for the Ollama API.
    """
    print("\n--- Starting Interactive Chat Session ---")
    print(f"Chatbot is powered by {llm_model}. Ask me anything about {KANDA_NAME}!")
    print("Type 'exit' or 'quit' to end the session.")

    # Initialize the Ollama LLM
    try:
        llm = Ollama(model=llm_model, base_url=api_url)
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        print("Please ensure Ollama is running and Llama 3 is downloaded.")
        return

    # Define a prompt template that explicitly asks for citations
    template = """
    You are an expert on the Valmiki Ramayana. Your task is to answer the user's question based ONLY on the provided context. 
    Do not use any external knowledge. If the answer is not in the context, state that you cannot answer.
    
    For every factual claim you make, you MUST cite the original verse reference from the context. The reference is provided in the metadata of each document.
    Format your citations like this: [citation_id].

    Context:
    {context}
    
    Question: {question}

    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Set up the RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}), # Retrieve top 3 most relevant documents
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    while True:
        user_query = input("\n[You]: ")
        if user_query.lower() in ["exit", "quit"]:
            print("Ending session. Goodbye!")
            break

        # Get response from the RAG chain
        response = qa_chain.invoke({"query": user_query})
        
        # Format the output with the answer and citations
        answer = response.get("result", "Sorry, I could not find a relevant answer.")
        source_docs = response.get("source_documents", [])
        
        print(f"\n[Chatbot]: {answer}")
        
        if source_docs:
            print("\nSources:")
            for doc in source_docs:
                citation_id = doc.metadata.get("citation_id", "N/A")
                print(f"- {citation_id}")
                # Optional: Print the content for verification
                # print(f"  Content: {doc.page_content[:100]}...")
'''
if __name__ == "__main__":
    # Configuration for the full run
    FULL_DATASET_URL = "<https://raw.githubusercontent.com/AshuVj/Valmiki_Ramayan_Dataset/refs/heads/main/data/Valmiki_Ramayan_Shlokas.json>"
    LOCAL_FILE_PATH = os.path.join("data", "Valmiki_Ramayan_Shlokas.json")

    # We change the folder name to distinguish it from the small test
    FULL_VECTOR_STORE_PATH = "ramayana_full_faiss_index"

    # Load ALL data (pass None for kanda)
    ramayana_data = get_ramayana_data(FULL_DATASET_URL, LOCAL_FILE_PATH, kanda=None)

    if not ramayana_data:
        print("Exiting pipeline due to data retrieval error.")
    else:
        # Create or load the vector store with GPU acceleration
        vector_store = create_rag_vector_store(
            ramayana_data,
            FULL_VECTOR_STORE_PATH,
            EMBEDDING_MODEL_NAME
        )