# 🧠 Ramayana Semantic Verifier – IYD Hackathon

This project leverages **Large Language Models (LLMs)** and **semantic search** to verify whether a statement is true according to **Valmiki's Ramayana**. 

Basically a **A custom, lightweight RAG-style semantic verifier for Ramayana — purpose-built for high-accuracy fact checking using retrieval + generation.**


You can:
- 🖐️ Enter queries manually
- 📊 Upload a batch CSV file for bulk verification

---

## ✅ Already Done

- ✅ **Data has been extracted** from the original Valmiki Ramayana source  
- ✅ **Extracted data has been cleaned** and structured into `kanda`, `sarga`, `shloka`, and `explanation` format

---

## 📦 Setup Instructions

### 1️⃣ Install & Launch Ollama (to run the LLM in the background) and Pull `qwen3:14b`

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve > /dev/null 2>&1 &
ollama pull qwen3:14b
```

### 2️⃣ Install Required Python Packages

```bash
pip install gdown langchain langchain_community langchain_ollama sentence-transformers scikit-learn
```

---

## 🚀 How It Works

### 📥 Step 1: Download or Generate Embedded Dataset

You can:
- 🔽 Automatically download precomputed embeddings (via Google Drive), or
- 🧠 Generate your own embeddings using the provided `Encoder.py` script (see 📄 Dataset Embedding Script below)

---

### 🧠 Step 2: Semantic Search

For a given query:

- Generate its embedding using BERT (`distilbert-base-nli-stsb-mean-tokens`)
- Retrieve top N semantically similar Ramayana sentences using **cosine similarity** from `scikit-learn`

---

### 🤖 Step 3: LLM-based Fact Verification

Using the `qwen3:14b` model via **Ollama**, we prompt the LLM with:

- The user's query  
- Top semantically relevant sentences (with references)  
- A structured instruction asking it to return only:
  - `"True"` → if the query matches Ramayana facts
  - `"False"` → if the query contradicts Ramayana
  - `"None"` → if the query is unrelated to Ramayana

---

## ⚙️ Modes Supported

### 🖐️ Manual Mode

Type one query at a time and get a real-time verification result.

---

### 📊 Batch Mode

Upload a `.csv` file with the following format:

```csv
ID,Statement
1,"Ravana had 10 heads"
2,"Rama was from Hastinapur"
```

After processing, you’ll receive a downloadable CSV with the following structure:

| ID | Statement                | Predicted | Reference |
|----|--------------------------|-----------|-----------|
| 1  | Ravana had 10 heads      | True      | 6,43-12   |
| 2  | Rama was from Hastinapur | False     |           |

---

## 🔗 Dataset Embeddings

This tool depends on **precomputed embeddings and metadata** for fast semantic search.

- 📂 You can use the `Encoder.py` script (included in this repository) to generate your own embedding file (`complete_shloka_with_embeddings.json`) from the cleaned dataset (`ramayana.json`)

---

## 📄 Dataset Embedding Script – `Encoder.py`

You can generate your own embeddings by running:

```bash
python Encoder.py
```

This script will:
1. Load the cleaned Valmiki Ramayana data (`ramayana.json`)
2. Extract relevant fields: `kanda`, `sarga`, `shloka`, and `explanation`
3. Generate semantic embeddings using the model `distilbert-base-nli-stsb-mean-tokens`
4. Save the data with embeddings to `complete_shloka_with_embeddings.json`

---

## 💡 Technologies Used

| Tool/Library           | Purpose                            |
|------------------------|------------------------------------|
| `qwen3:14b` (Ollama)   | Local LLM inference                |
| SentenceTransformers   | Embedding generation               |
| scikit-learn           | Cosine similarity search           |
| LangChain + Ollama     | Prompt-based LLM interaction       |

---

## 🧪 Example Prompt

```
Query: Did Hanuman burn Lanka?

Sentences:
1. (5,42-12) Hanuman set Lanka ablaze with his burning tail.
2. (5,42-13) The city was consumed by flames started by Hanuman.

Answer:
True
```

---

## 🤝 Acknowledgements

- 💻 Built for **IYD Hackathon**
- 🔗 Powered by **Open-Source LLMs**, **LangChain**, and **Semantic Search**
-   **DATA Extraction Pipeline link** ---> https://github.com/KrishnaGupta0405/IYD-RAMAYAN-HACKATHON
