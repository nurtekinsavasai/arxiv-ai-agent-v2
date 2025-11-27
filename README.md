📘 Research Agent v2

A lightweight research assistant that fetches, ranks, and explains recent AI papers from arXiv.
Runs fully in Streamlit, supports OpenAI, Google Gemini, or a free local model, and produces ranked tables plus top-N highlight summaries.

Now includes Human–Computer Interaction (cs.HC) in addition to cs.AI and cs.LG.

🎥 Watch the short demo: https://youtu.be/4CvYLwlhXac

🚀 What's New in v2

Three pipeline modes

You can run the system using:

1. OpenAI Mode

Requires an OpenAI API key.
Enables the full experience:

LLM-based relevance classification

1-year citation impact scoring using LLM judgment

Plain English summaries of top papers

Explanation factors for impact scores

Higher-accuracy and more nuanced ranking

2. Gemini Mode (Google AI)

Requires a Google Gemini API key.
Provides the full experience using Google's latest models:

Supports Gemini 3 Pro (Preview), Gemini 2.5 Flash, Gemini 2.5 Pro, and Gemini 2.0 Flash (Exp)

Fast parallel processing for classification

Plain English summaries and citation impact scoring

3. Free Local Model Mode (Default)

Runs entirely locally on your machine.

No API key needed

Uses local embeddings (MiniLM-L6-v2)

Uses simple heuristic relevance + heuristic citation scoring

Skips LLM summaries

Still fetches and processes up to ~150 papers

Great for quick browsing or for users without API access

All modes share the same UI and the same pipeline.
Only the classification and scoring steps differ.

🧠 How It Works (Pipeline Overview)

Whether using OpenAI, Gemini, or free mode, the pipeline is:

1. You provide

A short research brief

Optional "not looking for" text

A date window: Last 3 days, Last week, Last month

2. The agent fetches papers

It downloads recent papers from these categories:

cs.AI – Artificial Intelligence

cs.LG – Machine Learning

cs.HC – Human–Computer Interaction

3. Candidate selection

In targeted mode, it embeds the brief and abstracts then selects top-150 most similar

In global mode, it simply takes the most recent 150

4. Relevance classification

OpenAI / Gemini mode: LLM assigns primary, secondary, or off-topic

Free mode: heuristic based on embedding similarity

5. Citation scoring set

The agent:

Keeps all primary papers

Tops up with the strongest secondary papers until reaching ~20, when possible

In global mode, uses all candidates

6. Citation impact scoring

OpenAI / Gemini mode: asks the LLM for a 1-year citation impact score and 3 explanation factors

Free mode: derives a score from relevance and embedding similarity

These are impact signals, not predictions.

7. Ranking

Primary papers ranked first by citation score

Secondary papers follow

Off-topic (rare) come last

8. Top-N highlighted papers

Displays:

Metadata

Links to PDF and arXiv

Relevance reasoning

Citation impact score

Plain English summary (OpenAI / Gemini mode only)

9. Export

A full Markdown report and all intermediate JSON files saved under:

~/arxiv_ai_digest_projects/project_<timestamp>


Downloadable as a ZIP from the UI.

💻 Running Locally (Only a UI — No backend server required)

Clone the repo:

git clone [https://github.com/nurtekinsavasai/arxiv-ai-agent-v2.git](https://github.com/nurtekinsavasai/arxiv-ai-agent-v2.git)
cd arxiv-ai-agent-v2


Create a virtual environment:

python3 -m venv .venv
source .venv/bin/activate


Install dependencies:

pip install -r requirements.txt
pip install google-genai  # Required if using Gemini mode


Run the app:

streamlit run app.py


Your browser opens automatically.

🔌 Choosing a Provider

Option A — OpenAI

Select "OpenAI" and enter:

API key

Chat model (default: gpt-4.1-mini)

Enables full classification, scoring, explanations, and summaries.

Option B — Gemini (Google AI)

Select "Gemini" and enter:

API Key (from Google AI Studio)

Chat model (default: gemini-3-pro-preview or gemini-2.5-flash)

Enables full classification, scoring, explanations, and summaries.

Option C — Free Local Model

Select "Free local model".

No API key needed

Uses CPU-only MiniLM embeddings

Uses heuristic relevance + heuristic citation impact
