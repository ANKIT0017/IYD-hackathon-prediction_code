import json
from sentence_transformers import SentenceTransformer

# Load the pretrained SBERT model
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

# Step 1: Load the dataset
def load_json_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Check if the file is empty or not
        if data is None or len(data) == 0:
            raise ValueError("The file is empty or contains no valid data.")
        return data
    except Exception as e:
        print(f"Error loading the file: {e}")
        return None

# Step 2: Extract relevant fields (kanda, sarga, shloka, explanation)
def extract_relevant_data(data):
    extracted_data = []
    for entry in data:
        # Ensure each entry is a dictionary and contains all necessary fields
        if isinstance(entry, dict) and all(key in entry for key in ['kanda', 'sarga', 'shloka', 'explanation']):
            # Check if explanation is not None or empty
            if entry['explanation'] not in [None, '']:
                extracted_data.append({
                    'kanda': entry['kanda'],
                    'sarga': entry['sarga'],
                    'shloka': entry['shloka'],
                    'explanation': entry['explanation']
                })
        else:
            print(f"Skipping invalid entry: {entry}")
    return extracted_data

# Step 3: Generate embeddings for the explanations
def generate_embeddings(explanations, model):
    try:
        embeddings = model.encode(explanations)
        return embeddings
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return []

# Step 4: Save the extracted data along with embeddings
def save_data_with_embeddings(extracted_data, embeddings, output_file):
    for i in range(len(extracted_data)):
        extracted_data[i]['embedding'] = embeddings[i].tolist()  # Add the embedding as a list (convert numpy array)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(extracted_data, f, ensure_ascii=False, indent=4)

# Main process
def process_data(file_path, output_file):
    data = load_json_data(file_path)          # Load the data from file
    if data is None:
        print("Failed to load data. Exiting.")
        return

    extracted_data = extract_relevant_data(data)  # Extract relevant fields
    if not extracted_data:
        print("No valid data found.")
        return

    # Extract explanations for embedding generation
    explanations = [entry['explanation'] for entry in extracted_data]

    embeddings = generate_embeddings(explanations, model)  # Generate embeddings

    # Check if embeddings are generated and have data
    if embeddings is not None and len(embeddings) > 0:  # Only save if embeddings were generated
        save_data_with_embeddings(extracted_data, embeddings, output_file)  # Save the results with embeddings
        print("Data processed, embeddings generated, and saved successfully!")
    else:
        print("Failed to generate embeddings.")

# Example usage:
file_path = 'ramayana.json'  # Replace with your file path (the file that contains your dataset)
output_file = 'complete_shloka_with_embeddings.json'
process_data(file_path, output_file)
