# Sakhii Bot 
**A RAG-based Healthcare Assistant developed for Sakhii Care Foundation**

## Features
- Hybrid BM25 + Semantic retrieval (MedQuAD dataset)
- Personalized responses based on user profile
- Multilingual support (English + Hinglish)
- Embedding-based emergency detection
- Conversation memory with auto-summarization
- Source attribution on every answer

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/jayasree1805/medical-rag-assistant.git
cd medical-rag-assistant
```

### 2. Create virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file:
gemini_token=your_key

### 5. Prepare the data
Add `medquad.csv` to the `data/` folder, then run:
```bash
python prepare_data.py
python -m src.retrieval
```

### 6. Run the bot
```bash
python test.py
```

### 7. Run evaluation
```bash
python eval.py
```

## Project Structure
src/
├── query.py          # Gemini API gateway + retry
├── emergency.py      # Embedding-based emergency detection
├── analyzer.py       # Translation + intent + query rewriting
├── retrieval.py      # Hybrid BM25 + semantic retrieval
├── memory.py         # Conversation history management
├── prompt.py         # Prompt construction + safety guard
└── user_profile.py   # User onboarding + language selection

## Tech Stack
- Google Gemini 2.5 Flash
- ChromaDB
- Sentence Transformers (MedEmbed-small-v0.1)
- BM25 (rank_bm25)
- MedQuAD Dataset

## Acknowledgements & Citations

### Embedding Model
This project uses **MedEmbed** for medical-domain semantic search and emergency detection.

```bibtex
@software{balachandran2024medembed,
  author = {Balachandran, Abhinand},
  title = {MedEmbed: Medical-Focused Embedding Models},
  year = {2024},
  url = {https://github.com/abhinand5/MedEmbed}
}
```

### Dataset
This project uses the **MedQuAD (Medical Question Answer Dataset)**
by Asma Ben Abacha and Dina Demner-Fushman (NIH/NLM).

```bibtex
@ARTICLE{BenAbacha-BMC-2019,
  author  = {Asma {Ben Abacha} and Dina Demner{-}Fushman},
  title   = {A Question-Entailment Approach to Question Answering},
  journal = {{BMC} Bioinform.},
  volume  = {20},
  number  = {1},
  pages   = {511:1--511:23},
  year    = {2019},
  url     = {https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-3119-4}
}
```

### LLM
Powered by **Google Gemini 2.5 Flash** via the Gemini API.