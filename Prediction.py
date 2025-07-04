import json, numpy as np, subprocess, time, re
import gdown
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from google.colab import files  # Optional for Colab
from httpx import ConnectError  # for error handling

# ─────────────────────────────────────────────────────────────
# Model Initialization
# ─────────────────────────────────────────────────────────────
embed_model    = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
llama_verifier = OllamaLLM(model="qwen3:14b")

verify_template = """
Forget previous conversations. You will carefully compare the query with meaning of combined list of valmiki-ramayana sentences given.
If any one of the sentences or their combined abstract meaning equals the in-depth meaning of the query or the query is a well-known fact from Ramayana, reply with "True" along with the reference kanda,sarga,shloka.
If it is not from Ramayana or does not match the meaning of the given sentences, or is factually incorrect per the Ramayana, reply "False".
If it is not related to Ramayana just reply "None"
You will not say anything else.

Query: {query}

Sentences (with kanda, sarga, shloka):
{sentences}

Answer:
"""
verify_prompt = ChatPromptTemplate.from_template(verify_template)

# ─────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────
def download_from_drive(file_id, output_name):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_name, quiet=False)


def load_with_embeddings(fp):
    with open(fp, 'r', encoding='utf-8') as f:
        return json.load(f)


def semantic_search(query, dataset, top_n=50):
    q_emb    = embed_model.encode([query])[0]
    all_embs = np.array([e['embedding'] for e in dataset])
    sims     = cosine_similarity([q_emb], all_embs)[0]
    idxs     = np.argsort(sims)[::-1][:top_n]
    return [
        (
            float(sims[i]),
            dataset[i]['kanda'],
            dataset[i]['sarga'],
            dataset[i]['shloka'],
            dataset[i]['explanation']
        )
        for i in idxs
    ]


def clean_response(text: str) -> str:
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def verify_with_llama(query, results, retries=1):
    formatted = "\n".join(
        f"{i+1}. ({k},{sarga}-{shloka}) {text}"
        for i, (_, k, sarga, shloka, text) in enumerate(results)
    )
    prompt = verify_prompt.format_prompt(
        query=query,
        sentences=formatted
    ).to_string()

    for attempt in range(retries + 1):
        try:
            raw = llama_verifier.invoke(prompt).strip()
            return clean_response(raw)
        except ConnectError:
            if attempt < retries:
                print("⚠️ Ollama not running. Starting server in background...")
                subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                time.sleep(5)
            else:
                return "Error: Ollama not available"
        except Exception as e:
            return f"Error: {e}"

# ─────────────────────────────────────────────────────────────
# Main Logic
# ─────────────────────────────────────────────────────────────
def main():
    DRIVE_FILE_ID = "1Gt056ULo9K7Bfgg90MrsfluOH8Ua8Tvi"  # <- Updated file ID
    OUTPUT_FILE   = "with_embeddings.json"

    print(" Downloading precomputed model embedding vectors ...")
    download_from_drive(DRIVE_FILE_ID, OUTPUT_FILE)

    print(" Loading embedded vectors...")
    data_with_emb = load_with_embeddings(OUTPUT_FILE)
    print(f" Loaded {len(data_with_emb)} vectors.\n")

    mode = input("Choose mode:\n1. Upload CSV file for batch processing\n2. Manual input mode\nEnter 1 or 2: ").strip()

    if mode == "1":
        print(" Please upload a CSV file with `ID,Statement` format.")
        uploaded = files.upload()
        file_name = list(uploaded.keys())[0]

        df = pd.read_csv(file_name)
        print("\nTop 10 rows for confirmation:")
        print(df.head(10).to_string(index=False))
        input("\nPress Enter to proceed with processing...")

        ids, statements, preds, refs = [], [], [], []

        for _, row in df.iterrows():
            q_id    = row['ID']
            query   = str(row['Statement'])
            results = semantic_search(query, data_with_emb, top_n=10)
            verdict = verify_with_llama(query, results)

            ref = ""
            if verdict.lower().startswith("true") and results:
                _, k, sarga, shloka, _ = results[0]
                ref = f"{k},{sarga}-{shloka}"

            print(f"Statement: {query}\nPrediction: {verdict}\nReference: {ref}\n{'-'*50}")

            ids.append(q_id)
            statements.append(query)
            preds.append(verdict)
            refs.append(ref)

        out_df = pd.DataFrame({
            "ID": ids,
            "Statement": statements,
            "Predicted": preds,
            "Reference": refs
        })

        output_name = "output_verification.csv"
        out_df.to_csv(output_name, index=False)
        print(f"✅ Saved output to `{output_name}`.")
        files.download(output_name)

    else:
        print("\n🔍 Ready for manual queries. Type 'exit' to quit.\n")
        while True:
            query = input("Your query: ").strip()
            if not query or query.lower() == 'exit':
                print("Goodbye!")
                break

            results = semantic_search(query, data_with_emb, top_n=20)
            verdict = verify_with_llama(query, results)
            print(f"\nVerification: {verdict}\n{'─'*60}\n")

if __name__ == "__main__":
    main()
