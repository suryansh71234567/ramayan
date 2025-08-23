import os
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

# Create Flask app instance
app = Flask(__name__)

# Global variables to store the vector store and embeddings
vector_store = None
embeddings = None
VECTOR_STORE_PATH = "sunderkand_faiss_index"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

# Prompt template for the LLM
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

def load_resources():
    """
    Loads the FAISS vector store and embedding model.
    This function is called once on application startup.
    """
    global vector_store, embeddings
    print("Loading embedding model...")
    embeddings = HuggingFaceBgeEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    print("Loading vector store from disk...")
    if not os.path.exists(VECTOR_STORE_PATH):
        print("Error: Vector store not found. Please build it first.")
        # In a real app, you would handle this more gracefully, perhaps by
        # raising an exception or returning a 500 error on API calls.
        return
    try:
        vector_store = FAISS.load_local(
            VECTOR_STORE_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        print("Vector store loaded successfully.")
    except Exception as e:
        print(f"Error loading vector store: {e}")
        vector_store = None


@app.route("/")
def home():
    """
    A simple route to check if the API is running.
    """
    return "The Valmiki Ramayana RAG API is running!"

@app.route("/chat", methods=["POST"])
def chat():
    """
    Main API endpoint for the chatbot.
    """
    # Check if the vector store is loaded
    if vector_store is None:
        return jsonify({"error": "Vector store is not available."}), 503

    # Get user query from the request body
    data = request.get_json()
    user_query = data.get("query")
    if not user_query:
        return jsonify({"error": "Missing 'query' parameter."}), 400

    try:
        # Retrieve relevant documents from the vector store
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        docs = retriever.get_relevant_documents(user_query)
        context_text = "\n\n".join([d.page_content for d in docs])
        
        # Prepare the prompt for the LLM
        formatted_prompt = prompt.format(context=context_text, question=user_query)

        # Prepare the API request payload for OpenRouter
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "mistralai/mistral-7b-instruct",
            "messages": [
                {"role": "user", "content": formatted_prompt}
            ]
        }
        
        # Make the API call
        resp = requests.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        answer = resp.json()["choices"][0]["message"]["content"]
        
        # Extract and prepare citations for the response
        sources = [d.metadata.get("citation_id", "N/A") for d in docs]

        # Return the response as JSON
        return jsonify({
            "answer": answer,
            "sources": sources
        })

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": f"An internal error occurred: {e}"}), 500

if __name__ == "__main__":
    # Load resources before the server starts
    load_resources()
    # Run the Flask app
    app.run(host="0.0.0.0", port=5000)