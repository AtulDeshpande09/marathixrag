# 🌐 MarathiXRAG

> Cross-lingual Retrieval-Augmented Generation (RAG) for Marathi queries using English and Hindi Wikipedia corpora.

MarathiXRAG enables users to ask questions in **Marathi** and retrieve semantically relevant knowledge chunks from **English** and **Hindi Wikipedia** using multilingual embeddings and translation-assisted retrieval.

---

## ✨ Features

* 🔍 Cross-lingual semantic retrieval
* 🇮🇳 Marathi → English/Hindi query translation
* 🧠 Multilingual embeddings using BGE-M3
* 📚 Wikipedia-based retrieval corpus
* ⚡ Fast vector search with ChromaDB
* 📈 High retrieval accuracy on benchmark queries

---

## 📊 Benchmark Results

Evaluation on **1,000 Marathi queries**.

| Top-K | Retrieval Accuracy |
| ----- | ------------------ |
| Top-1 | **88.6%**          |
| Top-3 | **93.0%**          |
| Top-5 | **93.9%**          |

---

# ⚡ Quick Start

## 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/MarathiXRAG.git
cd MarathiXRAG
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```


## 🚀 Run Demo

```bash
python src/retrieval/demo.py \
  --query "स्वीडनमध्ये डोनाल्ड बर्गागार्ड कोणत्या चळवळीचे संस्थापक बनले?"
```

Example Output:

```text
Top Retrieved Chunks:

1. Donald Bergagård (born 13 February 1936 in Öckerö) is a Swedish singer, accordionist and evangelist....
2. Bergagård wrote the Swedish text to the song där rosor...
```

---

# 🏗️ Architecture

```text
                 Wikipedia Corpus (EN/HI)
                           │
                           ▼
              [Qwen2.5-7B Question Generation]
                           │
                           ▼
Marathi Query → [Sarvam Translation]
                           │
                           ▼
                [BGE-M3 Embedding Model]
                           │
                           ▼
                  ChromaDB Vector Search
                           │
                           ▼
                    Top-K Relevant Chunks
```

---

# 🧩 Tech Stack

| Component           | Tool / Model              |
| ------------------- | ------------------------- |
| Translation         | Sarvam Translate API      |
| Embeddings          | BAAI/bge-m3               |
| Question Generation | Qwen2.5-7B-Instruct       |
| Vector Database     | ChromaDB                  |
| Dataset             | English & Hindi Wikipedia |
| Language            | Python                    |

---

# 📁 Project Structure

```text
MarathiXRAG/
│
├── data/                 # Processed datasets
├── notebooks/            # Experiments and evaluation
├── src/
│   ├── retrieval/        # Retrieval pipeline
│   ├── embedding/        # Embedding utilities
│   ├── translation/      # Translation modules
│   └── evaluation/       # Benchmark scripts
│
├── requirements.txt
├── README.md
└── .env
```

---

# 🧪 Evaluation

The system was evaluated using manually curated Marathi queries mapped to relevant English/Hindi Wikipedia passages.

Metrics used:

* Top-K Retrieval Accuracy
* Semantic Relevance Matching
* Cross-Lingual Retrieval Performance

---

# 🔮 Future Improvements

* Add reranking with multilingual cross-encoders
* Support additional Indian languages
* Hybrid BM25 + dense retrieval
* Fine-tuned Marathi embedding models
* Web interface for interactive querying

---

# 📜 License

This project is licensed under the MIT License.

---

# 🙌 Acknowledgements

* Sarvam AI
* Hugging Face
* Wikipedia
* BAAI

---
