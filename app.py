# app.py
import os
import itertools, time
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# --- RAG Specific Imports ---
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
# For direct API calls, we use the openai library
from openai import OpenAI

# --- Configuration for your LLM and RAG system ---
load_dotenv()

# --- API Key Management and Rotation Logic ---
# Add your OpenRouter API keys to this list
API_KEYS = [
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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"

]
# Remove any None values from the list in case some keys are not set
API_KEYS = [key for key in API_KEYS if key is not None]

# Check if at least one key is available
if not API_KEYS:
    print("Error: No OPENROUTER_API_KEY environment variables found.")
    exit()

# Set up key cycling and initial client
api_keys_cycle = itertools.cycle(API_KEYS)
current_key = next(api_keys_cycle)
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=current_key)


def call_with_key_rotation(model, messages, max_retries=10):
    """
    Calls the OpenRouter API with key rotation and retry logic.
    """
    global client, current_key
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages
            )
            return completion
        except Exception as e:
            print(f"[Key Error] Key {current_key[:12]}... failed: {e}")
            # Rotate to the next key
            current_key = next(api_keys_cycle)
            client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=current_key)
            print(f"[Key Switch] Switched to {current_key[:12]}...")
            time.sleep(2 ** attempt)  # Exponential backoff
    raise RuntimeError("All API keys exhausted or failed after multiple retries.")


# --- RAG Specific Configurations ---
VECTOR_STORE_PATH = "sunderkand_faiss_index"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
LLM_MODEL = "mistralai/mistral-7b-instruct"
KANDA_NAME = "Sundarakāṇḍa"


# --- Function to load your FAISS vector store ---
def load_vector_store():
    """
    Loads the pre-built FAISS vector store.
    """
    print("Attempting to load vector store from local directory...")
    try:
        embeddings = HuggingFaceBgeEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        vector_store = FAISS.load_local(
            VECTOR_STORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("Vector store loaded successfully.")
        return vector_store
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return None


# --- RAG Chain Setup (runs once on server startup) ---
print("Initializing Ramayana RAG Chatbot...")

# 1. Load the vector store
vector_store = load_vector_store()
if vector_store is None:
    print("Failed to initialize vector store. The server will not be able to answer questions.")
    vector_store_ready = False
else:
    vector_store_ready = True

# 2. Set up the retriever
if vector_store_ready:
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    print("Retriever initialized successfully.")
else:
    retriever = None

# 3. Define the prompt template
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
prompt_template = ChatPromptTemplate.from_template(template)

# --- Flask API Setup ---
app = Flask(__name__)
CORS(app)


@app.route("/chat", methods=["POST"])
def chat():
    """
    API endpoint for the chatbot.
    Receives a user query and returns a RAG-based response.
    """
    if not retriever:
        return jsonify({"message": "Chatbot is not ready. Please check backend logs."}), 503

    data = request.json
    user_query = data.get("user_message")

    if not user_query:
        return jsonify({"error": "Missing user message"}), 400

    try:
        # Step 1: Retrieve relevant documents
        source_docs = retriever.get_relevant_documents(user_query)
        context_text = "\n\n".join([doc.page_content for doc in source_docs])

        # Step 2: Format the prompt with context
        messages = [
            {"role": "user", "content": prompt_template.format(context=context_text, question=user_query)}
        ]

        # Step 3: Invoke the LLM with key rotation
        completion = call_with_key_rotation(model=LLM_MODEL, messages=messages)
        answer = completion.choices[0].message.content

        # Extract citations
        citations = [doc.metadata.get("citation_id", "N/A") for doc in source_docs]

        # Prepare the response to send back to the frontend
        return jsonify({
            "message": answer,
            "citations": citations
        })

    except RuntimeError as e:
        print(f"Error processing chat request: {e}")
        return jsonify({"error": "All API keys exhausted or failed."}), 500
    except Exception as e:
        print(f"Unexpected error: {e}")
        return jsonify({"error": "An unexpected error occurred. Please try again."}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
