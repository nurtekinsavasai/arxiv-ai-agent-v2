# 📘 Research Agent v3

A lightweight research assistant that fetches, ranks, and explains recent AI papers from arXiv.
Runs fully in Streamlit, supports OpenAI, Google Gemini, or a free local model, and produces ranked tables plus top-N highlight summaries.

Now includes **Advanced Venue Filtering** and coverage for **Human–Computer Interaction (cs.HC)** alongside cs.AI and cs.LG.

🎥 **Watch the demo:** https://youtu.be/4CvYLwlhXac

---

## 🚀 What's New in v3

### 🏷️ Smart Venue Filtering

You can now filter papers by publication venue without losing semantic quality:

- **Filter Types:** "All Conferences", "All Journals", or "Specific Venue"
- **Specific Selection:** Pick individual top-tier venues like NeurIPS, CVPR, ICML, Nature, or Science
- **Smart Pipeline:** The agent performs semantic search on all papers first to understand the landscape, and then applies your venue filter. This ensures the AI sees the full context of related work before narrowing down to your preferences

### 🛡️ Robust arXiv Fetching

- Improved rate-limiting logic to strictly adhere to arXiv's API policies (3-second delays, smart backoff)
- Prevents IP bans during large data fetches

---

## Three Pipeline Modes

You can run the system using:

### 1. OpenAI Mode

**Requires:** OpenAI API key

**Features:**
- LLM-based relevance classification (Primary/Secondary/Off-topic)
- 1-year citation impact scoring using LLM judgment
- Plain English summaries of top papers
- Explanation factors for citation scores
- Highest accuracy ranking

### 2. Gemini Mode (Google AI)

**Requires:** Google Gemini API key

**Features:**
- Full support for Gemini 3 Pro (Preview), Gemini 2.5 Flash, and Gemini 2.0 Flash
- Fast parallel processing for classification
- Plain English summaries and citation impact scoring
- Cost-effective high performance

### 3. Free Local Model Mode (Default)

**Requires:** No API key (runs on CPU)

**Features:**
- Uses local embeddings (MiniLM-L6-v2) for search
- Uses simple heuristic relevance + heuristic citation scoring
- Skips LLM summaries/explanations
- Great for quick browsing or offline use

---

## 🧠 How It Works (Pipeline Overview)

Whether using OpenAI, Gemini, or free mode, the pipeline follows these 9 steps:

### 1. You provide a Brief

- A short natural language research brief (e.g., "Agents that use tool use")
- Optional "not looking for" constraints
- Date range (Last 3 days, Last week, Last month)

### 2. Fetch Papers

- Downloads recent papers from **cs.AI** (Artificial Intelligence), **cs.LG** (Machine Learning), and **cs.HC** (Human-Computer Interaction)
- Respects arXiv API rate limits

### 3. Candidate Selection

- **Targeted Mode:** Embeds your brief and paper abstracts, then selects the top ~150 semantically similar papers
- **Global Mode:** Selects the most recent 150 papers overall

### 4. Venue Filtering (New in v3)

- If enabled, filters the candidates to keep only those from your selected venues (e.g., "NeurIPS only")
- Applied **after** embedding search to ensure the best semantic matches are found first

### 5. Relevance Classification

- **LLM Mode:** The AI reads each abstract and classifies it as Primary, Secondary, or Off-topic
- **Free Mode:** Uses cosine similarity thresholds

### 6. Citation Scoring Set

- Keeps all Primary papers
- Tops up with the strongest Secondary papers until reaching ~20 candidates (to save costs/time)

### 7. Citation Impact Scoring

- **LLM Mode:** Asks the model to predict a "1-year citation impact score" based on topic trendiness, author fame, and novelty
- **Free Mode:** Derives a score from semantic relevance

### 8. Ranking & Highlights

- Ranks papers by their Impact Score
- Displays metadata, PDF links, and plain English summaries for the Top N

### 9. Export

- Generates a full Markdown Report
- Saves all intermediate JSON data
- Available as a ZIP download directly from the UI

---

## 💻 Running Locally

This is a **UI-only application (Streamlit)**. No backend server is required.

### 1. Clone the repo:

```bash
git clone https://github.com/nurtekinsavasai/arxiv-ai-agent-v2.git
cd arxiv-ai-agent-v2
```

### 2. Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
# On Windows: .venv\Scripts\activate
```

### 3. Install dependencies:

```bash
pip install -r requirements.txt
pip install google-genai  # Required if using Gemini mode
```

### 4. Run the app:

```bash
streamlit run app.py
```

Your browser will open automatically at `http://localhost:8501`.

---

## 🔌 Choosing a Provider

| Option | Requirements | Best For |
|--------|-------------|----------|
| **OpenAI** | API Key | Highest quality summaries and reasoning. (GPT-4o, GPT-4.1) |
| **Gemini** | API Key (Google AI Studio) | Speed and large context windows. (Gemini 3 Pro, Flash 2.5) |
| **Free Local** | CPU | Privacy, offline use, and zero cost |
