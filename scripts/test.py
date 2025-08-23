
    
import json
import requests
import os
from collections import defaultdict

# --- Configuration ---
# Path to your local Ramayana JSON file
DATA_FILE_PATH = r"C:\Users\ssing\Desktop\my_ramayana_project\data\Valmiki_Ramayan_Shlokas.json"

# Ollama API endpoint
OLLAMA_URL = "http://localhost:11434/api/generate"

# Name of the Llama 3 model you have pulled in Ollama
OLLAMA_MODEL = "llama3:8b"
DATASET_URL = "https://raw.githubusercontent.com/AshuVj/Valmiki_Ramayan_Dataset/refs/heads/main/data/Valmiki_Ramayan_Shlokas.json"
DATA_FILE_PATH = os.path.join("data", "Valmiki_Ramayan_Shlokas.json")
KANDA_NAME = "Sundara Kanda"
VECTOR_STORE_PATH = "sunderkand_faiss_index"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"  # A powerful multilingual model for Sanskrit/English
OLLAMA_API_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3"
# --- Functions ---


def load_sunderkand_data(file_path):
    """
    Loads the JSON data and extracts all shlokas from the Sunder Kanda,
    grouping them by sarga.
    """
    print(f"Loading data from: {file_path}")
    if not os.path.exists(file_path):
        print("Error: File not found.")
        return None

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            full_data = json.load(f)
    except json.JSONDecodeError:
        print("Error: Could not decode JSON from the file. Please check the file's integrity.")
        return None

    sunderkand_sargas = defaultdict(list)
    found_sunderkand = False

    for item in full_data:
        if item.get('kanda') == 'Sundara Kanda':
            found_sunderkand = True
            sarga_index = item.get('sarga')
            if sarga_index is not None:
                sunderkand_sargas[sarga_index].append(item)
            else:
                print(f"Warning: Item with missing 'sarga' key found in Sunder Kanda: {item}")
    
    if not found_sunderkand:
        print("Error: 'Sunder Kanda' not found in the dataset. Please check the 'kanda' field names.")
        return None
    
    # Sort sargas by their index for ordered processing
    sarga_list = sorted(sunderkand_sargas.items())
    
    print(f"Successfully loaded and grouped {len(sarga_list)} sargas from Sunder Kanda.")
    return sarga_list

def segment_sargas_by_shloka_count(sarga_list, shloka_limit=250):
    """
    Segments the sargas into logical subparts based on shloka count.
    Ensures that a single sarga is never split between prompts.
    """
    subparts = []
    current_shloka_count = 0
    current_sargas_in_part = []
    
    for sarga_index, shlokas in sarga_list:
        shlokas_in_sarga = len(shlokas)
        
        # If adding the next sarga exceeds the limit, or if it's the first sarga
        if current_shloka_count + shlokas_in_sarga > shloka_limit and current_shloka_count > 0:
            # Finalize the current subpart
            subparts.append(current_sargas_in_part)
            
            # Start a new subpart with the current sarga
            current_sargas_in_part = [(sarga_index, shlokas)]
            current_shloka_count = shlokas_in_sarga
        else:
            # Add the sarga to the current subpart
            current_sargas_in_part.append((sarga_index, shlokas))
            current_shloka_count += shlokas_in_sarga

    # Add the last subpart after the loop finishes
    if current_sargas_in_part:
        subparts.append(current_sargas_in_part)
    print(f"Segmented into {len(subparts)} subparts based on shloka count.")
    return subparts

def generate_with_ollama(prompt):
    """
    Sends a prompt to the local Ollama server and returns the generated text.
    Includes specific error handling for debugging.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3, # A lower temperature for more deterministic output
        }
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60) # Increased timeout
        response.raise_for_status()
        result = response.json()
        return result['response'].strip()
    except requests.exceptions.ConnectionError as e:
        print(f"Connection Error: Is the Ollama server running? Details: {e}")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: An error occurred with the API request. Details: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def get_subpart_name(sargas_in_part):
    """
    Analyzes a chunk of sargas using Llama 3 to generate a thematic name.
    """
    summary_text = ""
    sarga_indices = [s[0] for s in sargas_in_part]
    start_sarga, end_sarga = min(sarga_indices), max(sarga_indices)
    
    for sarga_index, shlokas in sargas_in_part:
        explanations = [shloka.get('explanation', '') for shloka in shlokas]
        sarga_explanations = " ".join([exp for exp in explanations if exp])
        summary_text += f"Sarga {sarga_index}: {sarga_explanations}\n\n"
        
    # The prompt for Llama 3, asking only for a name and nothing else.
    prompt = f"""
    Based on the following summary of Sunderkand, provide a short, thematic name for this part of the story. Do not provide any other text, just the name.

    Content:
    {summary_text}

    Name:"""
    
    print(f"Requesting name for sargas {start_sarga}-{end_sarga} from Llama 3...")
    name = generate_with_ollama(prompt)
    
    # Clean up the response just in case the model adds extra text
    if name and '"' in name:
        name = name.split('"')[1]
    
    return name or f"Untitled Part {start_sarga}-{end_sarga}"

# --- Main Execution ---

