import json
import requests
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configuration for LLaMA 3
# Ensure you have a Hugging Face token and the model is downloaded locally or accessible.
# Replace with the path to your local LLaMA 3 model directory
model_path = "meta-llama/Llama-3-8B-Instruct" 

# Initialize the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

def get_sunderkand_data(url):
    """Fetches and parses the Sunderkand data from the JSON file."""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors
        data = response.json()
        
        for kanda in data:
            if kanda['kanda_name'] == 'Sunder Kanda':
                return kanda['sarga']  # Assuming 'sarga' key exists
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def summarize_sargas_with_llama(sargas):
    """
    Summarizes each sarga's content and groups them into subparts using LLaMA 3.
    This is the core logic.
    """
    sunderkand_content = ""
    for sarga in sargas:
        shlokas_text = " ".join([shloka['shloka_text'] for shloka in sarga['shlokas']])
        sunderkand_content += f"Sarga {sarga['sarga_id']}: {shlokas_text}\n\n"

    # Define the prompt for LLaMA 3
    # The prompt instructs the model to act as a literary analyst,
    # summarize the text, and identify thematic subparts.
    prompt = f"""
    You are a literary analyst specializing in ancient Indian epics. I will provide you with the text of the Sunderkand from the Valmiki Ramayana, which is divided into 68 sargas. Your task is to analyze the content, identify major thematic shifts, and segment the entire section into 4-6 smaller, logical subparts. 

    For each subpart, you must:
    1. Provide a concise, descriptive name derived from the content.
    2. Specify the total number of sargas in that subpart.

    Thematic shifts often occur at significant events like:
    - Hanuman's journey to Lanka
    - The search for Sita
    - The meeting with Sita in the Ashoka Vatika
    - Hanuman's actions of destruction
    - The final journey back to Rama

    Here is the content of the Sunderkand:
    {sunderkand_content}

    Please format your output as a list of JSON objects, where each object represents a subpart.
    Example format:
    [
      {{
        "subpart_name": "Name of the Subpart",
        "sarga_count": number_of_sargas
      }},
      ...
    ]
    """

    # Generate the response
    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **input_ids,
            max_new_tokens=500,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # The model's response needs to be parsed, as it will contain the entire prompt
    # and the generated JSON.
    try:
        json_start = response.find('[')
        json_end = response.rfind(']') + 1
        json_output = response[json_start:json_end]
        return json.loads(json_output)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Could not parse JSON output: {e}")
        print("Raw response from model:", response)
        return []

# Main execution logic
if __name__ == "__main__":
    url = "https://raw.githubusercontent.com/AshuVj/Valmiki_Ramayan_Dataset/main/data/Valmiki_Ramayan_Shlokas.json"
    sunderkand_sargas = get_sunderkand_data(url)
    
    if sunderkand_sargas:
        print("Analyzing Sunderkand with LLaMA 3...")
        subparts = summarize_sargas_with_llama(sunderkand_sargas)
        
        print("\n--- Sunderkand Subparts ---")
        for i, part in enumerate(subparts):
            print(f"Subpart {i+1}:")
            print(f"  Name: {part['subpart_name']}")
            print(f"  Number of Sargas: {part['sarga_count']}")
            print("-" * 25)
    else:
        print("Failed to retrieve Sunderkand data. Exiting.")