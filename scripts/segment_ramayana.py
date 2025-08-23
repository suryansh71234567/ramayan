import json
import requests
import os

# --- Configuration ---
# Path to your local Ramayana JSON file
DATA_FILE_PATH = r"C:\Users\ssing\Desktop\my_ramayana_project\data\Valmiki_Ramayan_Shlokas.json"
print(f"Checking for file at: {DATA_FILE_PATH}")
if os.path.exists(DATA_FILE_PATH):
    print("File found.")
else:
    print("File not found.")
# Ollama API endpoint
OLLAMA_URL = "http://localhost:11434/api/generate"

# Name of the Llama 3 model you have pulled in Ollama
OLLAMA_MODEL = "llama3:8b "

# --- Functions ---

def load_sunderkand_data(file_path):
    """
    Loads the JSON data and extracts all shlokas from the Sunder Kanda.
    The function groups the shlokas by sarga.
    """
    if not os.path.exists(file_path):
        print(f"Error: The file at {file_path} was not found.")
        return None

    with open(file_path, 'r', encoding='utf-8') as f:
        full_data = json.load(f)
    
    sunderkand_data = [item for item in full_data if item.get('kanda') == 'Sunder Kanda']
    
    # Group shlokas by sarga
    sargas = {}
    for shloka in sunderkand_data:
        sarga_index = shloka.get('sarga')
        if sarga_index not in sargas:
            sargas[sarga_index] = []
        sargas[sarga_index].append(shloka)

    return sargas

def generate_with_ollama(prompt):
    """
    Sends a prompt to the local Ollama server and returns the generated text.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,  # We want a single, complete response
        "options": {
            "temperature": 0.3,
        }
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        return result['response']
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Ollama: {e}")
        return None

def segment_sunderkand(sargas_data):
    """
    Analyzes the sargas using Llama 3 to segment Sunderkand into subparts.
    """
    sunderkand_summary = ""
    sarga_ids = sorted(sargas_data.keys())
    
    for sarga_id in sarga_ids:
        explanations = [shloka.get('explanation', '') for shloka in sargas_data[sarga_id]]
        # We'll use the 'explanation' field as it's a good summary
        sarga_explanations = " ".join([exp for exp in explanations if exp])
        sunderkand_summary += f"Sarga {sarga_id}: {sarga_explanations}\n\n"

    # The prompt for Llama 3, guiding it to identify themes and segment the story
    prompt = f"""
    Based on the following summaries of the sargas from the Sunder Kanda of Valmiki Ramayana,
    segment the entire story into 4 to 6 logical subparts. For each subpart,
    provide a thematic name and the total number of sargas it contains.

    Thematic shifts often occur at major events such as:
    - Hanuman's journey and entry into Lanka.
    - The search for Sita and their first meeting.
    - Hanuman's actions of valor and destruction (e.g., meeting Akshakumara, burning Lanka).
    - Hanuman's return journey and reporting to Rama.

    The content for analysis is:
    {sunderkand_summary}

    Please provide the output as a clean JSON list, like this:
    [
      {{
        "subpart_name": "Hanuman's Journey to Lanka",
        "sarga_count": 5
      }},
      ...
    ]
    """

    print("Sending request to Ollama for analysis...")
    response_text = generate_with_ollama(prompt)
    
    if response_text:
        try:
            # Find and parse the JSON part of the response
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1
            json_str = response_text[json_start:json_end]
            return json.loads(json_str)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Could not parse JSON output from Llama 3: {e}")
            print("Raw response:", response_text)
            return []
    else:
        return []

# --- Main execution ---
if __name__ == "__main__":
    sargas_data = load_sunderkand_data(DATA_FILE_PATH)
    
    if sargas_data:
        print("Data loaded successfully. Starting Sunderkand segmentation.")
        segments = segment_sunderkand(sargas_data)
        
        if segments:
            print("\n--- Sunderkand Subpart Breakdown ---")
            for i, part in enumerate(segments, 1):
                print(f"Part {i}: {part.get('subpart_name')}")
                print(f"   Sarga Count: {part.get('sarga_count')}")
                print("-" * 30)
        else:
            print("Segmentation failed. Please check the Ollama server and model.")
    else:
        print("Exiting.")