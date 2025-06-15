# VoiceGrid: Neural Call Intelligence Engine

Welcome to the **VoiceGrid**, a complete solution for analyzing call recordings with AI-powered tools. This app makes it easy to find, review, and summarize voice interactions.

---

## 🚀 Project Overview

* **Semantic Search**: Find relevant calls based on meaning, not just keywords.
* **Filters**: Narrow results by date, speaker role, or call direction.
* **Playback & Download**: Play recordings directly in the browser or download audio files.
* **AI Summaries**: Get concise summaries and sentiment reports using GPT-4, DeepSeek, or Gemini.

**Data Pipeline**:

1. Fetch call data from the 8x8 API.
2. Transcribe audio with Whisper.
3. Create embeddings with OpenAI and store them in Pinecone.
4. Use Streamlit for an interactive search interface.
5. Apply AI models for deeper analysis.

---

## ✨ Key Features

* **Meaning-Based Search**: Locate calls that match your search intent.
* **Custom Filters**: Filter by metadata such as date, participants, and call type.
* **Integrated Audio Tools**: Play or batch-download recordings with a click.
* **AI-Powered Insights**: Extract key entities, sentiment, and summaries via GPT-4, DeepSeek, or Google Gemini.
* **Historical Data Import**: Run scripts to load past call data in bulk.

---

## 🛠 Tech Stack

* **Language**: Python 3.10+
* **Web App**: Streamlit
* **Transcription**: Whisper (OpenAI)
* **Vector Search**: OpenAI embeddings + Pinecone
* **AI Analysis**: GPT-4, DeepSeek API, Google Gemini
* **Call Source**: 8x8 API (OAuth)

---

## 📁 Project Structure

```plaintext
├── app.py                # Main Streamlit application
├── data_ingest.py        # Scripts for downloading and transcribing calls
├── embed_pipeline.py     # Embedding creation and database updates
├── last_six_months.py    # Batch import of historical calls
├── gemini_app.py         # Streamlit app using Google Gemini
├── credentials.py        # API keys and secrets (gitignored)
└── README.md             # Project documentation
```

---

## ⚙️ Setup & Installation

1. **Clone the repo**:

   ```bash
   git clone https://github.com/aqilsaboor/voicegrid.git
   cd voicegrid
   ```
2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```
3. **Configure credentials**:

   * Set `OPENAI_KEY`, `PINECONE_KEY`, `DEEPSEEK_KEY`
   * Set `CLIENT_ID`, `CLIENT_SECRET` for the 8x8 API
   * Set `GOOGLE_KEY` for Gemini
4. **Launch the app**:

   ```bash
   streamlit run app.py
   ```

---

## ▶️ Import Historical Data

```bash
python last_six_months.py
```

---

## 🎨 Architecture Overview

```text
8x8 API (calls)
  └─> OAuth authentication
  └─> Transcription with Whisper
  └─> Embedding generation (OpenAI)
  └─> Vector storage (Pinecone)
  └─> Streamlit UI & AI tools
```

This workflow uses AI services to help you quickly search call transcripts, review key points, and generate summaries.

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request to add features, fix bugs, or improve documentation.

---
