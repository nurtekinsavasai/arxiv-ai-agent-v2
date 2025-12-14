# 🔎 Research Agent v3 (with Groq Support)

A lightweight, intelligent research assistant that fetches, ranks, and explains recent AI papers from arXiv.

It runs fully in Streamlit and acts as a **"Universal Brain"** for research—you can plug in OpenAI, Google Gemini, or Groq to handle the reasoning, or run it entirely offline with a Free Local Model.

It produces ranked tables, citation impact scores, and "Plain English" summaries for the top papers.

🎥 **Watch the demo:** https://youtu.be/4CvYLwlhXac

---

## 🚀 What's New

### ⚡ Groq Integration (Llama 3.1)

You can now use **Groq** as your intelligence provider. This allows you to run high-performance open-source models (like Llama 3.1 70B, Mixtral 8x7b) with blazing fast inference speeds and **free API access**.

**Note:** In Groq mode, the agent uses local embeddings for search and the Groq LLM for reasoning.

### 🏷️ Smart Venue Filtering

Filter papers by publication venue without losing semantic quality:

- **Filter Types:** "All Conferences", "All Journals", or "Specific Venue"
- **Specific Selection:** Pick individual top-tier venues like NeurIPS, CVPR, ICML, Nature, or Science
- **Smart Pipeline:** The agent performs semantic search on all papers first to understand the landscape, and then applies your venue filter. This ensures the AI sees the full context of related work before narrowing down.

### 🛡️ Robust arXiv Fetching

- Improved rate-limiting logic to strictly adhere to arXiv's API policies
- Prevents IP bans during large data fetches by using smart backoff (3-second delays)

---

## 🔌 Four Intelligence Modes

You can choose your **"Brain"** from the sidebar:

### 1. OpenAI Mode (Best Quality)

**Requires:** OpenAI API Key

**Models:** GPT-4o, GPT-4.1, O1

**Best For:** Highest quality reasoning, summaries, and citation predictions. Uses OpenAI embeddings for search.

### 2. Gemini Mode (Best Context/Price)

**Requires:** Google Gemini API Key

**Models:** Gemini 3 Pro (Preview), Gemini 2.5 Flash, Gemini 2.0 Flash

**Best For:** Processing large batches of papers quickly. Uses Gemini embeddings.

### 3. Groq Mode (Fastest / Open Source)

**Requires:** Groq API Key (Currently Free)

**Models:** Llama 3.1 70B, Llama 3.1 8B, Mixtral 8x7b, Gemma 2

**Best For:** Extremely fast speed and using open-weight models.

**Architecture:** Uses Local Embeddings (MiniLM) for search + Groq for classification/scoring.

### 4. Free Local Mode (Offline / Privacy)

**Requires:** No API Key. Runs on CPU.

**Models:** SentenceTransformers (MiniLM-L6-v2) + Heuristics

**Best For:** Offline use, zero cost, and quick browsing without LLM summaries.

---

## 🧠 How It Works (The Pipeline)

The agent follows a **9-step pipeline** to distill hundreds of papers into the top 3:

1. **Briefing:** You provide a natural language "Research Brief" (e.g., "Agents that use tool use") and optional constraints.

2. **Fetching:** It downloads recent metadata from **cs.AI**, **cs.LG**, and **cs.HC** via the arXiv API.

3. **Semantic Search:**
   - **OpenAI/Gemini:** Uses cloud embeddings to find papers semantically similar to your brief.
   - **Groq/Local:** Uses local sentence-transformers to embed and rank papers on your CPU.

4. **Venue Filtering:** Filters candidates (e.g., "NeurIPS only") after the semantic search.

5. **LLM Classification:** The chosen LLM reads the abstracts and classifies papers as **Primary** (direct match), **Secondary** (partial match), or **Off-topic**.

6. **Scoring Set:** It builds a final list of ~20 papers to score, prioritizing Primary matches.

7. **Impact Scoring:** The LLM predicts a "1-Year Citation Impact Score" (0-100) based on novelty, author/lab reputation (if visible), and topic trendiness.

8. **Ranking:** Papers are ranked by this score.

9. **Reporting:** The UI displays metadata, PDF links, and generates a Plain English Summary for the top N papers.

---

## 💻 Usage Guide

### Option A: Run Locally (Recommended)

This is a pure Python application. No Docker or complex databases required.

#### 1. Clone the repository:

```bash
git clone https://github.com/nurtekinsavasai/arxiv-ai-agent-v2.git
cd arxiv-ai-agent-v2
```

#### 2. Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate
```

#### 3. Install dependencies:

```bash
pip install -r requirements.txt
# Ensure you install groq if it wasn't automatically picked up
pip install groq
```

#### 4. Run the app:

```bash
streamlit run app.py
```

Your browser will open automatically at `http://localhost:8501`.

### Option B: Deploy Online

You can deploy this repo directly to **Streamlit Community Cloud** for free:

1. Fork this repo to your GitHub.
2. Log in to [share.streamlit.io](https://share.streamlit.io).
3. Click "New App" and select your forked repo.

**Important:** In the Streamlit Cloud settings, go to "Secrets" and add your API keys if you want them pre-loaded, or just enter them in the UI sidebar every time you use it.

---

## 🛠️ For Developers

- **`app.py`:** The entire application logic resides here. It is a single-file Streamlit app designed for portability.

- **Modifying Prompt Logic:** Look for `classify_papers_with_llm` and `predict_citations_direct` functions in `app.py` to change how the AI judges papers.

- **Adding Providers:** The code uses a `LLMConfig` dataclass. To add a new provider (e.g., Anthropic), add the client initialization in `call_llm` and the specific embedding logic in `select_embedding_candidates`.

---

## 📊 Comparison Table

| Option | Requirements | Best For |
|--------|-------------|----------|
| **OpenAI** | API Key | Highest quality summaries and reasoning. (GPT-4o, GPT-4.1) |
| **Gemini** | API Key (Google AI Studio) | Speed and large context windows. (Gemini 3 Pro, Flash 2.5) |
| **Groq** | API Key (Free) | Fastest inference with open-source models. (Llama 3.1, Mixtral) |
| **Free Local** | CPU | Privacy, offline use, and zero cost |
