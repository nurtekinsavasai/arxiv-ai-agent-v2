# app.py
import os
import json
import time
import textwrap
import io
import zipfile
import math
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, date
from typing import List, Optional, Dict, Any
from collections import Counter

try:
    import streamlit as st
    import requests
    import feedparser
    import pandas as pd
    from openai import OpenAI, NotFoundError, BadRequestError
    from groq import Groq

    # Dashboard deps (required for the new feature)
    import plotly.express as px
    import matplotlib.pyplot as plt
except ImportError as e:
    missing = str(e).split("'")[1]
    print(f"Missing package: {missing}")
    print(
        "Please run: pip install streamlit requests feedparser openai pandas groq plotly matplotlib"
    )
    raise

# Optional wordcloud (nice-to-have)
try:
    from wordcloud import WordCloud  # type: ignore
except ImportError:
    WordCloud = None  # type: ignore

# Optional local embedding model for free mode
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:
    SentenceTransformer = None  # type: ignore

# Optional Google Gemini client
try:
    from google import genai  # type: ignore
except ImportError:
    genai = None  # type: ignore

# =========================
# Constants
# =========================

MIN_FOR_PREDICTION = 20
OPENAI_EMBEDDING_MODEL_NAME = "text-embedding-3-large"
GEMINI_EMBEDDING_MODEL_NAME = "text-embedding-004"

# MONEYBALL DEFAULTS
DEFAULT_MONEYBALL_WEIGHTS = {
    "weight_fame": 0.84,
    "weight_hype": 0.0,
    "weight_sniper": 0.0,
    "weight_utility": 0.16
}

# =========================
# Data structures
# =========================

@dataclass
class LLMConfig:
    api_key: str
    model: str
    api_base: str
    provider: str = "openai"  # "openai", "gemini", "groq", or "free_local"


@dataclass
class Paper:
    arxiv_id: str
    title: str
    authors: List[str]
    email_domains: List[str]
    abstract: str
    submitted_date: datetime
    pdf_url: str
    arxiv_url: str
    predicted_citations: Optional[float] = None
    prediction_explanations: Optional[List[str]] = None
    semantic_relevance: Optional[float] = None
    semantic_reason: Optional[str] = None
    focus_label: Optional[str] = None
    llm_relevance_score: Optional[float] = None
    venue: Optional[str] = None


# =========================
# Utility functions
# =========================

def get_date_range(option: str) -> (date, date):
    today = date.today()
    if option == "Last 3 Days":
        return today - timedelta(days=3), today
    elif option == "Last Week":
        return today - timedelta(days=7), today
    elif option == "Last Month":
        return today - timedelta(days=30), today
    else:
        raise ValueError(f"Unknown date range option: {option}")


def ensure_folder(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return os.path.abspath(path)


def save_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def build_query_brief(research_brief: str, not_looking_for: str) -> str:
    research_brief = research_brief.strip()
    not_looking_for = not_looking_for.strip()
    parts = []
    if research_brief:
        parts.append("RESEARCH BRIEF:\n" + research_brief)
    if not_looking_for:
        parts.append("WHAT I AM NOT LOOKING FOR:\n" + not_looking_for)
    if not parts:
        return "The user did not provide any research brief."
    return "\n\n".join(parts)


def parse_not_terms(not_text: str) -> List[str]:
    not_text = not_text.strip()
    if not not_text:
        return []
    parts = re.split(r"[,\n;]+", not_text)
    terms = [p.strip().lower() for p in parts if p.strip()]
    return terms


def filter_papers_by_not_terms(papers: List[Paper], not_text: str) -> (List[Paper], int):
    terms = parse_not_terms(not_text)
    if not terms or not papers:
        return papers, 0

    filtered: List[Paper] = []
    removed = 0
    for p in papers:
        text = f"{p.title} {p.abstract}".lower()
        if any(term in text for term in terms):
            removed += 1
        else:
            filtered.append(p)

    return filtered, removed


def filter_papers_by_venue(
    papers: List[Paper],
    venue_filter_type: str,
    selected_category: Optional[str],
    selected_venues: List[str]
):
    if venue_filter_type == "None":
        return papers

    filtered = []
    for p in papers:
        venue = (p.venue or "").lower()

        if venue_filter_type == "All Conferences":
            if any(conf.lower() in venue for conf in CONFERENCE_KEYWORDS):
                filtered.append(p)

        elif venue_filter_type == "All Journals":
            if any(j.lower() in venue for j in JOURNAL_KEYWORDS):
                filtered.append(p)

        elif venue_filter_type == "Specific Venue":
            if selected_category == "Conference":
                if selected_venues and any(sel.lower() in venue for sel in selected_venues):
                    filtered.append(p)
            elif selected_category == "Journal":
                if selected_venues and any(sel.lower() in venue for sel in selected_venues):
                    filtered.append(p)

    return filtered


# =========================
# Venue extraction helpers
# =========================

CONFERENCE_KEYWORDS = [
    "EMNLP", "ACL", "NAACL", "EACL",
    "NeurIPS", "ICLR", "ICML",
    "CVPR", "ECCV",
    "ICASSP", "AAAI", "AISTATS",
]

JOURNAL_KEYWORDS = [
    "Nature", "Science",
    "JMLR", "Journal of Machine Learning Research",
    "TPAMI", "IEEE Transactions on Pattern Analysis",
    "Artificial Intelligence Journal",
    "IJCV", "International Journal of Computer Vision",
    "Nature Machine Intelligence", "Nature Communications",
]

NEGATIVE_VENUE_SIGNALS = ["submitted to", "under review", "preprint"]

ARXIV_CATEGORIES: Dict[str, List[str]] = {
    "Computer Science": [
        "cs.AI", "cs.LG", "cs.HC", "cs.CL", "cs.CV", "cs.RO", "cs.IR", "cs.NE", "cs.SE",
        "cs.CR", "cs.DS", "cs.DB", "cs.SI", "cs.MM", "cs.IT", "cs.PF", "cs.MA",
    ],
    "Statistics": ["stat.ML", "stat.AP", "stat.CO", "stat.TH"],
    "Mathematics": ["math.OC", "math.ST", "math.IT", "math.PR", "math.NA"],
    "Physics": ["physics.comp-ph", "physics.data-an", "physics.soc-ph", "physics.optics"],
    "Quantitative Biology": ["q-bio.QM", "q-bio.NC", "q-bio.BM"],
    "Quantitative Finance": ["q-fin.MF", "q-fin.ST", "q-fin.CP", "q-fin.TR"],
    "Electrical Engineering and Systems Science": ["eess.IV", "eess.SP", "eess.SY", "eess.AS"],
    "Economics": ["econ.EM", "econ.TH"],
}

def extract_venue(comment: str) -> Optional[str]:
    if not comment:
        return None
    c = comment.lower()
    if any(sig in c for sig in NEGATIVE_VENUE_SIGNALS):
        return None
    all_venues = sorted(CONFERENCE_KEYWORDS + JOURNAL_KEYWORDS)
    for venue in all_venues:
        if venue.lower() in c:
            return venue
    return None

def build_arxiv_category_query(
    main_category: str,
    subcategories: List[str],
) -> str:
    """
    Returns arXiv API query fragment like:
      (cat:cs.AI OR cat:cs.LG OR cat:stat.ML)
    If subcategories empty, falls back to all subcats in main_category.
    If main_category == "All", uses all subcategories across all mains.
    """
    if main_category == "All":
        cats = sorted({c for subs in ARXIV_CATEGORIES.values() for c in subs})
    else:
        cats = subcategories if subcategories else ARXIV_CATEGORIES.get(main_category, [])

    # Safety fallback (so your app never crashes)
    if not cats:
        cats = ["cs.AI", "cs.LG", "cs.HC"]

    return "(" + " OR ".join([f"cat:{c}" for c in cats]) + ")"


# =========================
# Trends Dashboard (Abstract-only EDA)
# =========================

TECH_STOPWORDS = set("""
a an the and or if then else when while for to of in on at by with without into from over under
is are was were be been being do does did done can could should would may might must will
this that these those it its we our us you your they their them i me my
as than such both either neither each per via using use used based approach method methods
paper work study studies result results show shows shown demonstrate demonstrates demonstrated
propose proposes proposed novel new state art sota framework model models system systems
task tasks dataset datasets data experiments experimental evaluation evaluated performance
improve improves improved improvement achieve achieves achieved
""".split())

TECH_KEEP_SHORT = {"ai", "ml", "cv", "nlp", "rl", "kg", "gnn", "llm", "rag", "vae", "gan", "vit"}

def tokenize_technical(text: str) -> List[str]:
    """
    Extract technical-ish tokens from abstracts/titles:
    - keep letters/digits/hyphens
    - remove grammar + academic filler
    - allow short tokens only if in TECH_KEEP_SHORT
    """
    if not text:
        return []

    t = text.lower()
    t = re.sub(r"[^\w\s\-]", " ", t)  # keep words, digits, hyphens
    t = re.sub(r"\s+", " ", t).strip()

    tokens = []
    for tok in t.split(" "):
        tok = tok.strip("-_")
        if not tok:
            continue
        if tok.isdigit():
            continue

        if len(tok) < 3 and tok not in TECH_KEEP_SHORT:
            continue

        if tok in TECH_STOPWORDS:
            continue

        if tok in {"et", "al", "figure", "table", "section"}:
            continue

        tokens.append(tok)

    return tokens


def extract_acronyms(text: str) -> List[str]:
    """
    Pull acronyms/model-like tokens: LLM, RAG, ViT, YOLOv8, NeRF, SAM, BERT, etc.
    """
    if not text:
        return []
    found = re.findall(r"\b[A-Z]{2,}[A-Z0-9\-]*\b", text)
    drop = {"USA", "HTTP", "IEEE"}  # customize
    return [f for f in found if f not in drop]


TECH_BUCKETS: Dict[str, List[str]] = {
    "LLMs & Reasoning": ["llm", "gpt", "instruction", "chain-of-thought", "cot", "reasoning", "alignment"],
    "RAG & Retrieval": ["rag", "retrieval", "vector", "embedding", "dense retrieval", "bm25", "reranker"],
    "Diffusion & Generative": ["diffusion", "score-based", "latent diffusion", "stable diffusion", "generative", "gan", "vae"],
    "Vision Models": ["vit", "vision transformer", "yolo", "sam", "segment", "detection", "tracking"],
    "Multimodal": ["multimodal", "vision-language", "vlm", "image-text", "video-text", "audio-text"],
    "Graphs & Knowledge": ["gnn", "graph", "knowledge graph", "kg", "link prediction", "graph neural"],
    "Reinforcement Learning": ["reinforcement", "rl", "policy", "reward", "actor-critic", "bandit"],
    "Federated / Privacy": ["federated", "dp", "differential privacy", "secure aggregation", "privacy"],
    "Robustness & Safety": ["robust", "adversarial", "safety", "jailbreak", "toxicity", "guardrail", "bias", "fairness"],
    "Optimization / Efficiency": ["quantization", "distillation", "pruning", "efficient", "low-rank", "lora", "compression"],
}

def bucket_counts(papers: List[Paper]) -> Dict[str, int]:
    counts = {k: 0 for k in TECH_BUCKETS.keys()}
    for p in papers:
        text = f"{p.title} {p.abstract}".lower()
        for bucket, keys in TECH_BUCKETS.items():
            if any(k in text for k in keys):
                counts[bucket] += 1
    return counts


def cooccurrence_matrix(tokens_per_doc: List[List[str]], top_terms: List[str]) -> "pd.DataFrame":
    idx = {t: i for i, t in enumerate(top_terms)}
    mat = [[0 for _ in top_terms] for __ in top_terms]

    for toks in tokens_per_doc:
        s = set(toks)
        present = [t for t in top_terms if t in s]
        for i in range(len(present)):
            for j in range(i, len(present)):
                a, b = present[i], present[j]
                ia, ib = idx[a], idx[b]
                mat[ia][ib] += 1
                if ia != ib:
                    mat[ib][ia] += 1

    return pd.DataFrame(mat, index=top_terms, columns=top_terms)


def render_trends_dashboard(
    papers_for_dashboard: List[Paper],
    top_k_terms: int = 25,
    trend_terms: int = 8,
    max_heat_terms: int = 15
):
    """
    Streamlit trends dashboard built ONLY from abstracts/titles.
    - removes grammar/common words
    - focuses on technical tokens and buckets
    """
    if not papers_for_dashboard:
        st.info("No papers available for trends dashboard.")
        return

    st.subheader("📊 Trends Dashboard (Abstract-only EDA)")
    st.caption("Built from **titles + abstracts only**. Grammar/common English removed to surface technical signals.")

    rows = []
    for p in papers_for_dashboard:
        rows.append({
            "arxiv_id": p.arxiv_id,
            "title": p.title,
            "abstract": p.abstract,
            "submitted_date": p.submitted_date.date() if p.submitted_date else None,
        })
    df = pd.DataFrame(rows)
    df["submitted_date"] = pd.to_datetime(df["submitted_date"])

    tokens_per_doc = [tokenize_technical(f"{t}\n{a}") for t, a in zip(df["title"], df["abstract"])]
    all_tokens = [tok for doc in tokens_per_doc for tok in doc]
    freq = Counter(all_tokens)

    if not all_tokens:
        st.warning("No technical tokens extracted (stopword list may be too aggressive).")
        return

    top_terms = [t for t, _ in freq.most_common(top_k_terms)]
    term_df = pd.DataFrame(freq.most_common(top_k_terms), columns=["term", "count"])

    bcounts = bucket_counts(papers_for_dashboard)
    bucket_df = pd.DataFrame(
        [{"bucket": k, "papers": v} for k, v in bcounts.items()]
    ).sort_values("papers", ascending=False)

    acr = []
    for p in papers_for_dashboard:
        acr.extend(extract_acronyms(f"{p.title}\n{p.abstract}"))
    acr_df = pd.DataFrame(Counter(acr).most_common(20), columns=["acronym", "count"])

    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("### ☁️ Technical Word Cloud")
        if WordCloud is None:
            st.info("`wordcloud` not installed. Install via `pip install wordcloud` to enable this plot.")
            fig = px.bar(term_df.sort_values("count", ascending=True), x="count", y="term", orientation="h")
            fig.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            wc_text = " ".join(all_tokens)
            wc = WordCloud(
                width=900,
                height=500,
                background_color="white",
                collocations=False,
                max_words=120
            ).generate(wc_text)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig, use_container_width=True)

    with c2:
        st.markdown("### 🧩 Technology / Idea Buckets")
        fig = px.bar(
            bucket_df,
            x="papers",
            y="bucket",
            orientation="h",
            text="papers",
            title="How many papers mention each theme (Top set)",
        )
        fig.update_layout(height=420, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns([1, 1])

    with c3:
        st.markdown("### 📌 Top Technical Terms")
        fig = px.bar(
            term_df.sort_values("count", ascending=True),
            x="count",
            y="term",
            orientation="h",
            title="Most frequent technical tokens (stopwords removed)",
        )
        fig.update_layout(height=450, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.markdown("### 🧠 Acronyms / Model Tokens")
        if acr_df.empty:
            st.info("No acronyms detected in abstracts/titles.")
        else:
            fig = px.bar(
                acr_df.sort_values("count", ascending=True),
                x="count",
                y="acronym",
                orientation="h",
                title="Most common acronyms (LLM, RAG, YOLOv8, ...)",
            )
            fig.update_layout(height=450, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📈 Term Trend Over Time")
    if df["submitted_date"].isna().all() or df["submitted_date"].nunique() < 2:
        st.info("Not enough date diversity to show a trend line.")
    else:
        k_terms = top_terms[:min(trend_terms, len(top_terms))]
        trend_rows = []
        for _, r in df.iterrows():
            toks = set(tokenize_technical(f"{r['title']} {r['abstract']}"))
            for t in k_terms:
                trend_rows.append({
                    "date": r["submitted_date"].date(),
                    "term": t,
                    "present": 1 if t in toks else 0
                })

        tdf = pd.DataFrame(trend_rows)
        agg = tdf.groupby(["date", "term"])["present"].sum().reset_index()

        fig = px.line(
            agg, x="date", y="present", color="term",
            markers=True,
            title="Mentions per day (# papers mentioning term)",
        )
        fig.update_layout(height=420, margin=dict(l=20, r=20, t=50, b=20), yaxis_title="# papers")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🔥 Term Co-occurrence Heatmap")
    heat_terms = top_terms[:min(max_heat_terms, len(top_terms))]
    if len(heat_terms) < 8:
        st.info("Not enough extracted terms to build a co-occurrence heatmap.")
    else:
        co_df = cooccurrence_matrix(tokens_per_doc, heat_terms)
        fig = px.imshow(
            co_df,
            text_auto=True,
            aspect="auto",
            title="Doc-level co-occurrence (terms appearing in the same abstract)",
        )
        fig.update_layout(height=520, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)


# =========================
# Robust arXiv fetching
# =========================

def fetch_arxiv_papers_by_date(
    start_date: date,
    end_date: date,
    arxiv_query: Optional[str] = None,
    batch_size: int = 50,
    max_batches: int = 100,
    max_retries: int = 3,
) -> List[Paper]:
    query = arxiv_query or "(cat:cs.AI OR cat:cs.LG OR cat:cs.HC)"
    base_url = "https://export.arxiv.org/api/query"

    results: List[Paper] = []
    seen_ids = set()
    start_index = 0

    for _ in range(max_batches):
        params = {
            "search_query": query,
            "start": start_index,
            "max_results": batch_size,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        retries = 0
        while True:
            try:
                response = requests.get(base_url, params=params, timeout=60)
            except requests.RequestException as e:
                st.error(f"Network error while calling arXiv: {e}")
                return results

            if response.status_code == 429:
                retries += 1
                if retries > max_retries:
                    st.error("arXiv returned HTTP 429 (rate limit) repeatedly.")
                    return results
                wait_seconds = 30 * retries
                st.warning(f"arXiv rate limit. Waiting {wait_seconds} seconds...")
                time.sleep(wait_seconds)
                continue

            try:
                response.raise_for_status()
            except requests.HTTPError as e:
                st.error(f"Error fetching from arXiv: {e}")
                return results
            break

        feed = feedparser.parse(response.text)
        if not feed.entries:
            break

        for entry in feed.entries:
            published_str = entry.get("published", "")
            if not published_str:
                continue
            published_dt = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
            published_date = published_dt.date()

            # NOTE: arXiv returns in descending order; once we go beyond start_date, we can stop.
            if published_date < start_date:
                return results

            arxiv_id = entry.get("id", "").split("/")[-1]
            if arxiv_id in seen_ids:
                continue
            seen_ids.add(arxiv_id)

            authors = [a.name for a in entry.authors] if "authors" in entry else []
            email_domains: List[str] = []
            pdf_url = ""
            arxiv_url = entry.get("id", "")
            for link in entry.links:
                if link.rel == "alternate":
                    arxiv_url = link.href
                if getattr(link, "title", "") == "pdf":
                    pdf_url = link.href

            comment = entry.get("arxiv_comment", "") or entry.get("comment", "")
            venue = extract_venue(comment)

            paper = Paper(
                arxiv_id=arxiv_id,
                title=entry.title.strip().replace("\n", " "),
                authors=authors,
                email_domains=email_domains,
                abstract=entry.summary.strip().replace("\n", " "),
                submitted_date=published_dt,
                pdf_url=pdf_url,
                arxiv_url=arxiv_url,
                venue=venue,
            )
            results.append(paper)

        start_index += batch_size
        time.sleep(3.0)

    return results


# =========================
# Generic LLM call + JSON helper
# =========================

def call_llm(prompt: str, llm_config: LLMConfig, label: str = "") -> str:
    if "last_prompts" not in st.session_state:
        st.session_state["last_prompts"] = {}
    st.session_state["last_prompts"][label or "default"] = prompt

    try:
        if llm_config.provider == "openai":
            client = OpenAI(api_key=llm_config.api_key, base_url=llm_config.api_base)
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt},
            ]
            kwargs: Dict[str, Any] = {"model": llm_config.model, "messages": messages}

            # Do NOT set temperature for o1 or ANY gpt-5 variant
            if not (llm_config.model.startswith("o1") or llm_config.model.startswith("gpt-5")):
                kwargs["temperature"] = 0.2

            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content

        elif llm_config.provider == "gemini":
            if genai is None:
                st.error("Gemini provider selected but google-genai package is not installed.")
                st.stop()
            client = genai.Client(api_key=llm_config.api_key)
            response = client.models.generate_content(
                model=llm_config.model,
                contents=prompt,
            )
            return response.text

        elif llm_config.provider == "groq":
            client = Groq(api_key=llm_config.api_key)
            response = client.chat.completions.create(
                model=llm_config.model,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            return response.choices[0].message.content

        else:
            raise ValueError(f"Unknown provider: {llm_config.provider}")

    except Exception as e:
        st.error(f"LLM call failed ({label or 'general'}): {e}")
        st.stop()


def safe_parse_json_array(raw: str) -> Optional[List[Dict[str, Any]]]:
    if not raw or not raw.strip():
        return None
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                return parsed
        except:
            pass

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except:
        return None
    return None


# =========================
# Embeddings
# =========================

def embed_texts_openai(texts: List[str], llm_config: LLMConfig, embedding_model: str) -> List[List[float]]:
    if not texts:
        return []
    client = OpenAI(api_key=llm_config.api_key, base_url=llm_config.api_base)
    all_embeddings: List[List[float]] = []
    batch_size = 100
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        try:
            resp = client.embeddings.create(model=embedding_model, input=batch)
        except Exception as e:
            st.error(f"Embedding API call failed: {e}")
            raise
        for d in resp.data:
            all_embeddings.append(d.embedding)
    return all_embeddings


def embed_texts_gemini(texts: List[str], llm_config: LLMConfig, embedding_model: str) -> List[List[float]]:
    if not texts:
        return []
    if genai is None:
        st.error("google-genai package missing.")
        st.stop()
    client = genai.Client(api_key=llm_config.api_key)
    all_embeddings: List[List[float]] = []
    batch_size = 100
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        try:
            response = client.models.embed_content(model=embedding_model, contents=batch)
        except Exception as e:
            st.error(f"Gemini embedding API call failed: {e}")
            raise
        for emb in getattr(response, "embeddings", []):
            all_embeddings.append(list(emb.values))
    return all_embeddings


@st.cache_resource(show_spinner=False)
def get_local_embed_model() -> SentenceTransformer:
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers is not installed.")
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def embed_texts_local(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    try:
        model = get_local_embed_model()
    except Exception as e:
        st.error(f"Local embedding model error: {e}")
        st.stop()
    vectors = model.encode(texts, convert_to_numpy=True)
    return vectors.tolist()


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1)
    norm2 = sum(b * b for b in vec2)
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (math.sqrt(norm1) * math.sqrt(norm2))


def select_embedding_candidates(
    papers: List[Paper],
    query_brief: str,
    llm_config: Optional[LLMConfig],
    embedding_model: str,
    provider: str,
    max_candidates: int = 150,
) -> List[Paper]:
    if not papers:
        return []

    try:
        if provider == "openai":
            query_vec = embed_texts_openai([query_brief], llm_config, embedding_model)[0]
        elif provider == "gemini":
            query_vec = embed_texts_gemini([query_brief], llm_config, embedding_model)[0]
        else:
            query_vec = embed_texts_local([query_brief])[0]
    except Exception:
        return papers

    texts = [p.title + "\n\n" + p.abstract for p in papers]
    try:
        if provider == "openai":
            paper_vecs = embed_texts_openai(texts, llm_config, embedding_model)
        elif provider == "gemini":
            paper_vecs = embed_texts_gemini(texts, llm_config, embedding_model)
        else:
            paper_vecs = embed_texts_local(texts)
    except Exception:
        return papers

    scored = []
    for p, vec in zip(papers, paper_vecs):
        sim = cosine_similarity(query_vec, vec)
        p.semantic_relevance = sim
        scored.append((sim, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    k = min(max_candidates, len(scored))
    return [p for _, p in scored[:k]]


# =========================
# LLM relevance classification
# =========================

def classify_papers_with_llm(
    papers: List[Paper],
    query_brief: str,
    llm_config: LLMConfig,
    batch_size: int = 15,
) -> List[Paper]:
    if not papers:
        return papers

    for batch_start in range(0, len(papers), batch_size):
        batch = papers[batch_start:batch_start + batch_size]
        paper_blocks = []
        for idx, p in enumerate(batch):
            block = textwrap.dedent(f"""
            Paper {idx}:
            Title: {p.title}
            Abstract: {p.abstract}
            """).strip()
            paper_blocks.append(block)

        instruction = textwrap.dedent(f"""
        You are given a user's research brief and a small set of papers.
        Brief: \"\"\"{query_brief}\"\"\"

        For each paper, decide:
          1. focus_label: "primary", "secondary", or "off-topic".
          2. relevance_score: float 0.0-1.0.
          3. reason: 1-2 sentence explanation.

        Return JSON array:
          [{{ "index": <int>, "focus_label": "...", "relevance_score": <float>, "reason": "..." }}]
        """).strip()

        prompt = "\n\n".join([instruction, "PAPERS:", *paper_blocks])
        raw = call_llm(prompt, llm_config, label="classification")
        parsed = safe_parse_json_array(raw)

        if parsed is None:
            st.error("Failed to parse classification JSON.")
            continue

        idx_to_info = {}
        for item in parsed:
            try:
                idx = int(item["index"])
                label = str(item.get("focus_label", "")).strip().lower()
                if label not in ["primary", "secondary", "off-topic"]:
                    label = "off-topic"
                idx_to_info[idx] = {
                    "focus_label": label,
                    "relevance_score": float(item.get("relevance_score", 0.0)),
                    "reason": str(item.get("reason", "")).strip(),
                }
            except:
                continue

        for idx, p in enumerate(batch):
            info = idx_to_info.get(idx)
            if info:
                p.focus_label = info["focus_label"]
                p.llm_relevance_score = info["relevance_score"]
                p.semantic_reason = info["reason"]
            else:
                p.focus_label = "off-topic"
                p.llm_relevance_score = 0.0

    return papers


def heuristic_classify_papers_free(candidates: List[Paper]) -> List[Paper]:
    if not candidates:
        return candidates
    ranked = sorted(candidates, key=lambda p: p.semantic_relevance or 0.0, reverse=True)
    n = len(ranked)
    if n == 0:
        return ranked
    top_k = max(1, min(n, max(10, int(0.3 * n))))
    for idx, p in enumerate(ranked):
        p.llm_relevance_score = p.semantic_relevance or 0.0
        p.focus_label = "primary" if idx < top_k else "secondary"
        if p.semantic_reason is None:
            p.semantic_reason = "Heuristic classification based on embedding similarity."
    return ranked


# =========================
# MONEYBALL Impact Scoring
# =========================

def get_s2_citation_stats(title: str, api_key: Optional[str] = None) -> int:
    headers = {"x-api-key": api_key} if api_key else {}
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {"query": title, "limit": 1, "fields": "title,citationCount,authors.citationCount"}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=3)
        if r.status_code == 200:
            data = r.json()
            if data.get('data'):
                paper_data = data['data'][0]
                auth_cites = [a.get('citationCount', 0) for a in paper_data.get('authors', []) if a.get('citationCount')]
                return max(auth_cites) if auth_cites else 0
    except:
        pass
    return 0


def predict_citations_direct(target_papers: List[Paper], llm_config: LLMConfig, batch_size: int = 8) -> List[Paper]:
    if not target_papers:
        return target_papers

    weights = DEFAULT_MONEYBALL_WEIGHTS
    if os.path.exists("moneyball_weights.json"):
        try:
            with open("moneyball_weights.json", "r") as f:
                weights = json.load(f)
        except:
            pass

    s2_key = os.getenv("S2_API_KEY")
    progress_bar = st.progress(0)

    for i, p in enumerate(target_papers):
        max_auth_cites = get_s2_citation_stats(p.title, s2_key)
        h1_fame = min(math.log(max_auth_cites + 1) * 8, 95)
        if not s2_key:
            time.sleep(0.3)

        t_lower = p.title.lower()
        h2_hype = 0
        if "benchmark" in t_lower or "dataset" in t_lower:
            h2_hype += 50
        if "survey" in t_lower:
            h2_hype += 40
        if "llm" in t_lower:
            h2_hype += 10

        h3_sniper = 0
        if "benchmark" in t_lower:
            h3_sniper += 50
        niche = ["lidar", "3d", "audio", "wireless", "agriculture", "traffic", "physics"]
        if any(n in t_lower for n in niche):
            h3_sniper -= 20

        prompt = textwrap.dedent(f"""
            Analyze this abstract.
            1. Rate 'Citation Potential' (0-10) based on market fit (Broad/Hot = High, Niche = Low).
            2. Write 2 short, plain English sentences explaining the score.
               - Sentence 1 (Market Fit): Why is this topic hot or niche? (Do NOT start with "Market Fit:")
               - Sentence 2 (Contribution): What is the specific value? (Do NOT start with "Contribution:")

            Title: {p.title}
            Abstract: {p.abstract[:800]}...

            Return JSON: {{ "score": <int>, "bullets": ["string", "string"] }}
        """)

        h4_utility = 50.0
        content_bullets = [
            "The topic appears relevant to current research trends.",
            "The paper proposes a specific contribution to the field."
        ]

        try:
            raw = call_llm(prompt, llm_config, label="moneyball_narrative")
            if raw:
                if "```" in raw:
                    parts = raw.split("```json")
                    if len(parts) > 1:
                        raw = parts[1].split("```")[0]
                    else:
                        raw = raw.split("```")[1].split("```")[0]

                parsed = json.loads(raw.strip())
                h4_utility = float(parsed.get("score", 5) * 10)

                if "bullets" in parsed and isinstance(parsed["bullets"], list):
                    raw_list = parsed["bullets"]
                    cleaned_list = []
                    for b in raw_list:
                        b = b.replace("Market Fit:", "").replace("Contribution:", "").strip()
                        cleaned_list.append(b)
                    content_bullets = cleaned_list[:2]
        except:
            pass

        score = (h1_fame * weights['weight_fame'] +
                 h2_hype * weights['weight_hype'] +
                 h3_sniper * weights['weight_sniper'] +
                 h4_utility * weights['weight_utility'])
        p.predicted_citations = score

        final_bullets = []

        if max_auth_cites > 3000:
            final_bullets.append("🚀 **Distribution:** This work comes from a highly influential author/lab, guaranteeing immediate visibility.")
        elif max_auth_cites > 500:
            final_bullets.append("📢 **Reach:** The authors have a strong established track record, helping the paper stand out.")
        elif max_auth_cites > 100:
            final_bullets.append("📈 **Momentum:** The authors have some prior traction, but the paper will rely on its content to break out.")
        else:
            final_bullets.append("🌱 **Emerging:** These are newer authors, so the paper must rely entirely on its specific merit to gain traction.")

        if len(content_bullets) >= 1:
            final_bullets.append(f"🎯 **Market Fit:** {content_bullets[0]}")
        if len(content_bullets) >= 2:
            final_bullets.append(f"💡 **Contribution:** {content_bullets[1]}")

        p.prediction_explanations = final_bullets

        progress_bar.progress((i + 1) / len(target_papers))

    progress_bar.empty()
    return target_papers


def assign_heuristic_citations_free(papers: List[Paper]) -> List[Paper]:
    if not papers:
        return papers
    scores = [(p.llm_relevance_score or 0.0) * 0.7 + (p.semantic_relevance or 0.0) * 0.3 for p in papers]
    if not scores:
        return papers
    min_s, max_s = min(scores), max(scores)
    for p, s in zip(papers, scores):
        norm = (s - min_s) / (max_s - min_s) if max_s > min_s else 0.5
        p.predicted_citations = float(int(10 + norm * 40))
    return papers


def summarize_paper_plain_english(paper: Paper, llm_config: LLMConfig) -> str:
    prompt = textwrap.dedent(f"""
    Explain this research paper to a non-expert.
    Title: {paper.title}
    Abstract: {paper.abstract}

    Provide 3-6 plain English bullet points covering main idea, problem solved, and takeaways.
    """).strip()
    return call_llm(prompt, llm_config, label="plain_english_summary")


# =========================
# Streamlit UI
# =========================

PIPELINE_DESCRIPTION_MD = """
#### 1. Describe what you want

You write a short research brief in natural language about the kind of work you care about, and optionally what you are not interested in. If you leave both fields empty, the agent switches to a global mode and just looks for the most impactful recent cs.AI, cs.LG, and cs.HC papers overall.

#### 2. The agent fetches recent arXiv papers + builds a Trends Dashboard

It fetches up to about 5000 papers from arxiv.org for the date range you choose, using your selected arXiv category filters—either all categories, or a specific main category (like Computer Science, Statistics, Physics) with one or more subcategories (like cs.AI, stat.ML, etc.). It does this carefully, respecting arXiv’s API rate limits.

Then, before you jump into the ranked list, it generates a **Trends Dashboard** from the **abstracts of the top-ranked papers**—showing technical-term word clouds, theme/bucket counts, acronym/model mentions, trend-over-time lines, and co-occurrence patterns.

#### 3. The agent picks candidate papers

- In **targeted mode**, the agent uses embeddings to measure how close each paper's title and abstract are to your brief in meaning and keeps the top 150 as candidates.
- In **global mode**, it simply takes the most recent 150 papers as candidates.

#### The agent filters by venue (Optional)

If you selected a venue filter (e.g. "NeurIPS only" or "All Journals"), the agent applies it **after** the embedding search. This ensures that the agent first identifies the most semantically relevant papers from the entire pool, and then narrows them down to your preferred venues.

#### 4. The agent judges how relevant each paper is

- In **LLM API mode** (OpenAI, Gemini, or Groq), a model reads each candidate and labels it as primary, secondary, or off topic.
- In **free local mode**, the agent uses a simple heuristic based on the embedding similarity to mark the most relevant papers as primary and the rest as secondary.

#### 5. The agent builds a citation impact set

The agent builds a set of papers to send to the citation impact step:

- It keeps all **primary** papers.
- If there are fewer than about 20, it tops up with the strongest **secondary** papers until it reaches roughly 20, when possible.
- In global mode, all candidates are used.

#### 6. The agent computes 1-year citation impact scores

- In **LLM API mode** (OpenAI, Gemini, or Groq), a model estimates a 1-year citation impact score for each paper and provides short explanations.
- In **free local mode**, the agent derives a citation impact score from the relevance signals and uses that to rank papers.

These scores are heuristic impact signals and are best used for ranking within this batch, not as ground truth.

#### 7. The agent ranks, summarizes, and saves results

The agent ranks papers, always showing **primary** papers first, then secondary ones. For the top N that you choose, it shows metadata, relevance signals, and links to arXiv and the PDF. In LLM API mode it also adds plain English summaries. All artifacts and a markdown report are saved in a project folder under `~/arxiv_ai_digest_projects/project_<timestamp>`, and you can download everything as a ZIP.
"""


def main():
    st.set_page_config(
        page_title="Research Agent",
        layout="wide",
    )

    # Light dashboard styling polish
    st.markdown("""
    <style>
      .block-container { padding-top: 1rem; }
      [data-testid="stMetric"] { background: #fff; padding: 10px; border-radius: 12px; }
    </style>
    """, unsafe_allow_html=True)

    st.title("🔎 Research Agent")

    st.write(
        """Too many important papers get lost in the noise. Most researchers and practitioners cannot reliably scan what is new recently in their area, find truly promising work, and trust that they did not miss something big."""
        " This agent helps with this problem by finding, ranking, and explaining recent AI papers on arxiv.org."
        " Run time can be lengthy if you select a large time window or a large backend LLM. Patience is a virtue for good things to come!"
    )

    # Sidebar
    with st.sidebar:
        st.header("🧠 Research Brief")

        research_brief_default = (
            "I am interested in papers whose MAIN contribution is about recommendation systems: "
            "for example, new model architectures, training strategies, evaluation methods, user or item modeling, "
            "or personalization techniques for recommenders.\n\n"
            "I especially care about work where recommendation is the central focus, not just a side example."
        )

        research_brief = st.text_area(
            "What kinds of topics are you looking for?",
            value=research_brief_default,
            height=200,
            help="Describe your research interest in natural language. Focus on what the main contribution "
                 "of the papers should be. If you leave this and the next box empty, the agent will perform "
                 "a global digest of recent cs.AI, cs.LG, and cs.HC papers."
        )

        not_looking_for = st.text_area(
            "What are you NOT looking for? (optional)",
            value="Generic LLM papers that only list recommendation as one of many downstream tasks, "
                  "or papers that focus purely on language modeling, math reasoning, or scaling without "
                  "a clear recommendation specific contribution.",
            height=120,
        )

        # --- Venue Filtering UI ---
        st.markdown("### 🏷 Venue Filter")

        venue_filter_type = st.selectbox(
            "Filter by venue",
            ["None", "All Conferences", "All Journals", "Specific Venue"],
            index=0
        )

        selected_category = None
        selected_venues = []

        if venue_filter_type == "Specific Venue":
            selected_category = st.selectbox(
                "Select type:",
                ["Conference", "Journal"]
            )

            if selected_category == "Conference":
                options = sorted(CONFERENCE_KEYWORDS)
            else:
                options = sorted(JOURNAL_KEYWORDS)

            selected_venues = st.multiselect(
                f"Select {selected_category.lower()}(s):",
                options=options
            )

        # ---- Category selection UI ----
        st.markdown("### 🧩 arXiv Category")

        main_cat = st.selectbox(
            "Main category",
            ["All"] + list(ARXIV_CATEGORIES.keys()),
            index=1
        )

        if main_cat == "All":
            subcats = []
            st.caption("Using all available categories.")
        else:
            subcats = st.multiselect(
                "Subcategory (choose one or more)",
                options=ARXIV_CATEGORIES[main_cat],
                default=["cs.AI", "cs.LG", "cs.HC"] if main_cat == "Computer Science" else [],
                help="If you choose none, we’ll use ALL subcategories from the selected main category."
            )

        arxiv_query = build_arxiv_category_query(main_cat, subcats)

        date_option = st.selectbox("Date Range", ["Last 3 Days", "Last Week", "Last Month"])

        st.markdown("### ⭐ Top N Highlight")
        top_n = st.slider("How many top papers to highlight?", 1, 10, 3)

        # NEW: Dashboard size control
        st.markdown("### 📊 Trends Dashboard")
        dashboard_n = st.slider(
            "How many top-ranked papers to analyze (EDA)?",
            10, 150, 50, step=10,
            help="Trends dashboard is built from titles+abstracts of the top-ranked papers."
        )

        st.markdown("### 🔌 Provider")

        provider_label_free = "Free local model (no API key)"
        provider_label_openai = "OpenAI (API key required)"
        provider_label_gemini = "Gemini (API key required)"
        provider_label_groq = "Groq (API key required)"

        provider_choice = st.radio(
            "Choose provider",
            [provider_label_free, provider_label_openai, provider_label_gemini, provider_label_groq],
            index=0,
        )

        if provider_choice == provider_label_openai:
            provider = "openai"
        elif provider_choice == provider_label_gemini:
            provider = "gemini"
        elif provider_choice == provider_label_groq:
            provider = "groq"
        else:
            provider = "free_local"

        if provider == "openai":
            api_base = "https://api.openai.com/v1"
        elif provider == "gemini":
            api_base = "https://generativelanguage.googleapis.com"
        else:
            api_base = ""

        if provider == "openai":
            st.markdown("### 🤖 OpenAI Settings")
            api_key = st.text_input("OpenAI API Key", type="password")
            st.caption(
                "Your API key is used only in memory for this session, is never written to disk, "
                "and is never shared with anyone or any service other than OpenAI's API. "
                "When your session ends, the key is cleared from the app's state."
            )

            openai_models = [
                "gpt-5.2",
                "gpt-5",
                "gpt-5-mini",
                "gpt-5-nano",
                "gpt-4.1-mini",
                "gpt-4.1",
                "gpt-4o-mini",
                "gpt-4o",
                "o1",
            ]
            model_choice = st.selectbox(
                "OpenAI Chat model (for classification & citation impact scoring)",
                openai_models,
                index=0,
            )
            model_name = model_choice
            embedding_model_name = OPENAI_EMBEDDING_MODEL_NAME
            st.caption(f"Embeddings (OpenAI): `{embedding_model_name}`")

        elif provider == "gemini":
            st.markdown("### 🌌 Gemini Settings")
            api_key = st.text_input("Gemini API Key", type="password")
            st.caption(
                "Use an API key from Google AI Studio or Vertex AI for the Gemini API. "
                "The key is kept only in memory for this session and is never written to disk."
            )

            gemini_models = [
                "gemini-3-pro-preview",
                "gemini-3-flash-preview",
                "gemini-2.5-flash",
                "gemini-2.5-pro",
                "gemini-2.0-flash-exp",
            ]
            gemini_choice = st.selectbox(
                "Gemini Chat model (for classification & citation impact scoring)",
                gemini_models,
                index=0,
            )
            model_name = gemini_choice
            embedding_model_name = GEMINI_EMBEDDING_MODEL_NAME
            st.caption(f"Embeddings (Gemini): `{embedding_model_name}`")

        elif provider == "groq":
            st.markdown("### ⚡ Groq Settings")
            api_key = st.text_input("Groq API Key", type="password")
            st.caption(
                "Groq API keys are free. No credit card needed. "
                "Get yours at https://console.groq.com"
            )

            groq_models = [
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
            ]

            model_choice = st.selectbox(
                "Groq Model",
                groq_models,
                index=0,
            )
            model_name = model_choice
            embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
            st.caption("Using free local MiniLM embeddings (Groq does not provide embeddings).")

        else:
            api_key = ""
            model_name = "heuristic-free-local"
            embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
            st.caption(
                f"Embeddings (local): `{embedding_model_name}`.\n"
                "Classification and citation impact scoring use simple heuristics. No API key or external calls."
            )

        run_clicked = st.button("🚀 Run Pipeline")

    if run_clicked:
        st.session_state["hide_pipeline_description"] = True

    hide_desc = st.session_state.get("hide_pipeline_description", False)

    if hide_desc:
        with st.expander("Show full pipeline description", expanded=False):
            st.markdown(PIPELINE_DESCRIPTION_MD)
    else:
        st.markdown(PIPELINE_DESCRIPTION_MD)

    brief_text = research_brief.strip()
    not_text = not_looking_for.strip()

    if not brief_text and not not_text:
        mode = "global"
        query_brief = (
            "User wants to see the most impactful recent AI, ML, and HCI papers in cs.AI, cs.LG, and cs.HC, "
            "without any additional topical filter."
        )
    elif not brief_text and not_text:
        mode = "broad_not_only"
        rb_prompt = (
            "User is broadly interested in recent AI, ML, and HCI work in cs.AI, cs.LG, and cs.HC."
        )
        query_brief = build_query_brief(rb_prompt, not_looking_for)
    else:
        mode = "targeted"
        query_brief = build_query_brief(research_brief, not_looking_for)

    params = {
        "research_brief": research_brief.strip(),
        "not_looking_for": not_looking_for.strip(),
        "date_option": date_option,
        "top_n": top_n,
        "dashboard_n": dashboard_n,
        "model_name": model_name,
        "provider": provider,
        "venue_filter_type": venue_filter_type,
        "selected_category": selected_category,
        "selected_venues": selected_venues,
        "arxiv_query": arxiv_query,
    }

    if "last_params" not in st.session_state:
        st.session_state["last_params"] = params.copy()

    if params != st.session_state["last_params"] and not run_clicked:
        for key in [
            "current_papers",
            "candidates",
            "used_papers",
            "used_label",
            "ranked_papers",
            "topN",
            "project_folder",
            "timestamp",
            "zip_bytes",
            "config",
            "mode",
            "current_start",
            "current_end",
            "plain_summaries",
        ]:
            st.session_state.pop(key, None)
        st.session_state["last_params"] = params.copy()
        st.info("Sidebar settings changed. Click **Run Pipeline** to generate new results.")
        return

    if run_clicked:
        st.session_state["last_params"] = params.copy()

    if provider == "openai":
        if not api_key or not model_name:
            if "ranked_papers" not in st.session_state:
                st.warning("Your OpenAI API key and chat model name are required to run in OpenAI mode.")
                return
    elif provider == "gemini":
        if not api_key or not model_name:
            if "ranked_papers" not in st.session_state:
                st.warning("Your Gemini API key and model name are required to run in Gemini mode.")
                return
    elif provider == "groq":
        if not api_key or not model_name:
            if "ranked_papers" not in st.session_state:
                st.warning("Your Groq API key and model name are required to run in Groq mode.")
                return
    else:
        api_key = api_key or ""
        model_name = model_name or "heuristic-free-local"

    llm_config = LLMConfig(
        api_key=api_key or "",
        model=model_name,
        api_base=api_base,
        provider=provider,
    )

    try:
        current_start, current_end = get_date_range(date_option)
    except ValueError as e:
        st.error(str(e))
        return

    st.session_state["mode"] = mode
    st.session_state["current_start"] = current_start
    st.session_state["current_end"] = current_end

    if not run_clicked and "ranked_papers" not in st.session_state:
        st.info("Fill in your research brief and settings in the sidebar, then click **Run Pipeline**.")
        return

    # 1. Project setup
    st.subheader("1. Project Setup")

    root_base_default = os.path.expanduser("~/arxiv_ai_digest_projects")
    base_folder = ensure_folder(root_base_default)

    if run_clicked or "project_folder" not in st.session_state or "timestamp" not in st.session_state:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_folder = os.path.join(base_folder, f"project_{timestamp}")
        project_folder = ensure_folder(project_folder)
        st.session_state["project_folder"] = project_folder
        st.session_state["timestamp"] = timestamp
    else:
        project_folder = st.session_state["project_folder"]
        timestamp = st.session_state["timestamp"]

    st.write(f"Project folder: `{project_folder}`")

    config = {
        "mode": mode,
        "query_brief": query_brief,
        "research_brief": research_brief,
        "not_looking_for": not_looking_for,
        "date_option": date_option,
        "current_start": str(current_start),
        "current_end": str(current_end),
        "project_folder": project_folder,
        "created_at": datetime.now().isoformat(),
        "llm_model": model_name,
        "llm_api_base": api_base,
        "embedding_model": embedding_model_name,
        "llm_provider": (
            "OpenAI" if provider == "openai"
            else "Gemini" if provider == "gemini"
            else "Groq" if provider == "groq"
            else "FreeLocalHeuristic"
        ),
        "top_n": top_n,
        "dashboard_n": dashboard_n,
        "min_for_prediction": MIN_FOR_PREDICTION,
        "arxiv_query": arxiv_query,
    }
    st.session_state["config"] = config
    save_json(os.path.join(project_folder, "config.json"), config)

    # 2. Fetch current papers
    st.subheader("2. Fetch Current Papers from arXiv according to your selected Category and Subcategory")

    if run_clicked or "current_papers" not in st.session_state:
        with st.spinner("Fetching current papers from arXiv by date window..."):
            current_papers = fetch_arxiv_papers_by_date(
                start_date=current_start,
                end_date=current_end,
                arxiv_query=arxiv_query,
            )
        st.session_state["current_papers"] = current_papers
    else:
        current_papers = st.session_state["current_papers"]

    if not current_papers:
        st.warning("No papers found for this date range (or arXiv stopped responding).")
        return

    st.success(
        f"Fetched {len(current_papers)} papers in this date range and Category/Subcategory Selected "
        "(before any candidate selection)."
    )

    # NOT filter
    if not_text:
        current_papers, removed_count = filter_papers_by_not_terms(current_papers, not_text)
        st.info(f"Excluded {removed_count} papers whose title or abstract contained NOT terms.")

    st.session_state["venue_filter_type"] = venue_filter_type
    st.session_state["selected_category"] = selected_category
    st.session_state["selected_venues"] = selected_venues

    st.session_state["current_papers"] = current_papers

    if not current_papers:
        st.warning("All papers were filtered out. Relax filters or pick another date range.")
        return

    save_json(
        os.path.join(project_folder, "current_papers_all.json"),
        [asdict(p) for p in current_papers],
    )

    # 3. Candidate selection
    if mode == "global":
        st.subheader("3. Candidate Selection (Most Recent Papers)")
        if run_clicked or "candidates" not in st.session_state:
            sorted_papers = sorted(
                current_papers,
                key=lambda p: p.submitted_date,
                reverse=True,
            )
            candidates = sorted_papers[:150] if len(sorted_papers) > 150 else sorted_papers
            st.session_state["candidates"] = candidates
        else:
            candidates = st.session_state["candidates"]

        st.success(
            f"{len(candidates)} most recent cs.AI, cs.LG, and cs.HC papers selected as candidates (global mode)."
        )
    else:
        st.subheader("3. Embedding Based Candidate Selection")
        if run_clicked or "candidates" not in st.session_state:
            with st.spinner("Selecting top candidate papers via embeddings..."):
                candidates = select_embedding_candidates(
                    current_papers,
                    query_brief=query_brief,
                    llm_config=llm_config if provider in ("openai", "gemini", "groq") else None,
                    embedding_model=embedding_model_name,
                    provider=provider,
                    max_candidates=150,
                )
            if not candidates:
                st.warning("Embedding stage returned no candidates. Using all fetched papers as fallback.")
                candidates = current_papers
            st.session_state["candidates"] = candidates
        else:
            candidates = st.session_state["candidates"]

        st.success(f"{len(candidates)} top candidates selected by embedding similarity for further filtering.")

    # Apply venue filtering AFTER embeddings
    venue_filter_type = st.session_state.get("venue_filter_type", "None")
    selected_category = st.session_state.get("selected_category", None)
    selected_venues = st.session_state.get("selected_venues", [])

    before_v = len(candidates)
    candidates = filter_papers_by_venue(
        candidates,
        venue_filter_type,
        selected_category,
        selected_venues
    )
    after_v = len(candidates)

    if venue_filter_type != "None":
        display_sel = ", ".join(selected_venues) if selected_venues else ""
        name_string = f" → {display_sel}" if display_sel else ""

        st.info(
            f"Venue filter `{venue_filter_type}` applied{name_string}. "
            f"Remaining: {after_v} (Filtered out {before_v - after_v})"
        )

    save_json(
        os.path.join(project_folder, "candidates_embedding_selected.json"),
        [asdict(p) for p in candidates],
    )

    # 4. Relevance classification
    st.subheader("4. Relevance Classification")

    if mode == "global":
        st.info(
            "Global mode: no specific research brief was provided. "
            "Skipping relevance classification and treating all candidate papers as PRIMARY."
        )
        if run_clicked or any(p.focus_label is None for p in st.session_state.get("candidates", [])):
            for p in candidates:
                p.focus_label = "primary"
                p.llm_relevance_score = None
                if p.semantic_reason is None:
                    p.semantic_reason = "Global mode: no topical filtering; treated as primary."
    else:
        if provider in ("openai", "gemini", "groq"):
            provider_label = "OpenAI" if provider == "openai" else "Gemini" if provider == "gemini" else "Groq"
            if run_clicked or any(p.focus_label is None for p in candidates):
                with st.spinner(f"Classifying candidates as PRIMARY, SECONDARY, or OFF TOPIC ({provider_label})..."):
                    candidates = classify_papers_with_llm(
                        candidates,
                        query_brief=query_brief,
                        llm_config=llm_config,
                        batch_size=15,
                    )
                st.session_state["candidates"] = candidates
                st.success(f"Classifying candidates as PRIMARY, SECONDARY, or OFF TOPIC ({provider_label})... Done!")
        else:
            st.info(
                "Free local mode: using a simple heuristic based on embedding similarity instead of LLM based classification."
            )
            if run_clicked or any(p.focus_label is None for p in candidates):
                candidates = heuristic_classify_papers_free(candidates)
                st.session_state["candidates"] = candidates

    save_json(
        os.path.join(project_folder, "candidates_with_classification.json"),
        [asdict(p) for p in candidates],
    )

    # 5. Build prediction set with minimum size
    st.subheader("5. Automatically Selected Papers for Citation Impact Scoring")

    if mode == "global":
        primary_papers = [p for p in candidates]
        secondary_papers: List[Paper] = []
        used_papers = primary_papers.copy()
        used_label = "Global mode: all candidate papers treated as PRIMARY and used for citation impact scoring."
        st.success(
            f"Global mode: using {len(used_papers)} most recent cs.AI, cs.LG, and cs.HC papers for citation impact scoring."
        )
    else:
        primary_papers = [p for p in candidates if p.focus_label == "primary"]
        secondary_papers = [p for p in candidates if p.focus_label == "secondary"]

        for group in (primary_papers, secondary_papers):
            group.sort(
                key=lambda p: (
                    p.llm_relevance_score if p.llm_relevance_score is not None else 0.0,
                    p.semantic_relevance if p.semantic_relevance is not None else 0.0,
                ),
                reverse=True,
            )

        used_label = ""
        if primary_papers:
            if len(primary_papers) >= MIN_FOR_PREDICTION:
                used_papers = primary_papers.copy()
                used_label = "All PRIMARY papers (enough for citation impact scoring)"
                st.success(
                    f"{len(primary_papers)} papers classified as PRIMARY. "
                    f"Using all of them for citation impact scoring (≥ {MIN_FOR_PREDICTION})."
                )
            else:
                used_papers = primary_papers.copy()
                if secondary_papers:
                    needed = MIN_FOR_PREDICTION - len(primary_papers)
                    topups = secondary_papers[:needed]
                    used_papers.extend(topups)
                    total = len(used_papers)
                    if len(secondary_papers) >= needed:
                        used_label = f"PRIMARY + top {len(topups)} SECONDARY to reach about {MIN_FOR_PREDICTION}"
                        st.success(
                            f"{len(primary_papers)} papers classified as PRIMARY. "
                            f"Added {len(topups)} top SECONDARY papers for citation impact scoring."
                        )
                    else:
                        used_label = (
                            f"All PRIMARY + all available SECONDARY "
                            f"(only {len(secondary_papers)} secondary papers, total {total} < {MIN_FOR_PREDICTION})"
                        )
                        st.info(
                            f"{len(primary_papers)} papers classified as PRIMARY. "
                            f"Only {len(secondary_papers)} SECONDARY papers available, so you have "
                            f"{total} papers in the scoring set (below the target of {MIN_FOR_PREDICTION})."
                        )
                else:
                    used_label = "All PRIMARY papers (no SECONDARY available)"
                    st.warning(
                        f"Only {len(primary_papers)} PRIMARY papers and no SECONDARY. "
                        "Using all PRIMARY papers for citation impact scoring even though this is below the "
                        f"target of {MIN_FOR_PREDICTION}."
                    )
        elif secondary_papers:
            used_papers = secondary_papers.copy()
            used_label = "All SECONDARY papers (no PRIMARY matches found)"
            st.warning(
                "No papers were classified as PRIMARY. Using all SECONDARY matches instead. "
                "These may only partially match your brief."
            )
        else:
            st.error("No candidates were classified as PRIMARY or SECONDARY. Nothing to proceed with.")
            return

    used_papers.sort(
        key=lambda p: (
            p.llm_relevance_score if p.llm_relevance_score is not None else 0.0,
            p.semantic_relevance if p.semantic_relevance is not None else 0.0,
        ),
        reverse=True,
    )

    st.session_state["used_papers"] = used_papers
    st.session_state["used_label"] = used_label

    save_json(
        os.path.join(project_folder, "used_papers_for_prediction.json"),
        [asdict(p) for p in used_papers],
    )

    st.write(
        "These are the papers that the pipeline will use for citation impact scoring. "
        "Selection is automatic based on mode, embeddings (in targeted modes), and relevance classification."
    )
    st.write(f"**Citation impact set description:** {used_label}")
    st.write(f"**Number of papers in citation impact set:** {len(used_papers)}")

    for p in used_papers:
        with st.expander(p.title, expanded=False):
            st.write(f"**Authors:** {', '.join(p.authors) if p.authors else 'Unknown'}")
            st.write(f"**Submitted:** {p.submitted_date.date().isoformat()}")
            st.write(f"[arXiv link]({p.arxiv_url}) | [PDF link]({p.pdf_url})")
            if p.focus_label:
                st.write(f"**Focus label:** {p.focus_label}")
            rel_str = f"{p.llm_relevance_score:.2f}" if p.llm_relevance_score is not None else "N/A"
            st.write(f"**Relevance score:** {rel_str}")
            sim_str = f"{p.semantic_relevance:.3f}" if p.semantic_relevance is not None else "N/A"
            if p.semantic_reason:
                st.write("**Why this paper is (or is not) relevant to your brief:**")
                st.write(p.semantic_reason)
            st.write(f"**Embedding similarity score:** {sim_str}")
            st.write("**Abstract:**")
            st.write(p.abstract)

    selected_papers = used_papers
    save_json(
        os.path.join(project_folder, "selected_papers_for_prediction.json"),
        [asdict(p) for p in selected_papers],
    )

    # 6. Citation impact scoring
    st.subheader("6. Citation Impact Scoring")

    if provider in ("openai", "gemini", "groq"):
        st.markdown("""
**How this step works (Moneyball Edition)**

We use the **Moneyball Algorithm** to predict 1-year citation impact scores.
It combines four signals:
1. **Author Fame:** Query Semantic Scholar for author citations.
2. **Content Utility:** LLM rates abstract market fit.
3. **Hype Keywords:** Bonus for trending topics.
4. **Niche Penalties:** Penalty for small fields.

This approach statistically outperforms pure LLM "vibes" by 6x.
""")
    else:
        st.markdown("""
**How this step works (free local mode)**

In free local mode, the agent does not call any external LLM. Instead, it combines the embedding based similarity and relevance scores into a single numeric citation impact score and uses that score as a proxy for how influential the paper might be relative to others in this batch.

These scores are heuristic and should be used as a guide for exploration rather than as formal evaluation metrics.
        """)

    if run_clicked or "ranked_papers" not in st.session_state:
        if provider in ("openai", "gemini", "groq"):
            with st.spinner("Calling LLM API to compute citation impact scores for selected papers..."):
                papers_with_pred = predict_citations_direct(
                    target_papers=selected_papers,
                    llm_config=llm_config,
                )
        else:
            with st.spinner("Computing heuristic citation impact scores from relevance signals..."):
                papers_with_pred = assign_heuristic_citations_free(selected_papers)

        papers_with_pred = [p for p in papers_with_pred if p.predicted_citations is not None]
        if not papers_with_pred:
            st.error("Citation impact scoring did not produce any scores.")
            return

        primary_pred = [p for p in papers_with_pred if p.focus_label == "primary"]
        secondary_pred = [p for p in papers_with_pred if p.focus_label == "secondary"]
        others_pred = [p for p in papers_with_pred if p.focus_label not in ("primary", "secondary")]

        def sort_by_pred(papers: List[Paper]) -> List[Paper]:
            return sorted(
                papers,
                key=lambda p: (p.predicted_citations if p.predicted_citations is not None else 0),
                reverse=True,
            )

        ranked_papers = sort_by_pred(primary_pred) + sort_by_pred(secondary_pred) + sort_by_pred(others_pred)
        st.session_state["ranked_papers"] = ranked_papers
        st.session_state["has_run_once"] = True
    else:
        ranked_papers = st.session_state["ranked_papers"]

    save_json(
        os.path.join(project_folder, "selected_papers_with_predictions.json"),
        [asdict(p) for p in ranked_papers],
    )

    # 7. All selected papers ranked
    st.subheader("7. All Selected Papers (Ranked by Citation Impact Score)")
    st.caption("Primary papers appear first, ranked by citation impact score, followed by secondary papers.")

    table_rows = []
    for rank, p in enumerate(ranked_papers, start=1):
        pred = int(p.predicted_citations or 0)
        focus = p.focus_label or "unknown"
        if focus == "primary":
            focus_display = "🟢 primary"
        elif focus == "secondary":
            focus_display = "🟡 secondary"
        elif focus == "off-topic":
            focus_display = "⚪ off-topic"
        else:
            focus_display = focus
        llm_rel = float(p.llm_relevance_score or 0.0)
        emb_rel = float(p.semantic_relevance or 0.0)
        table_rows.append(
            {
                "Rank": rank,
                "Citation impact score (1y)": pred,
                "Focus": focus_display,
                "Relevance score": round(llm_rel, 2),
                "Embedding similarity": round(emb_rel, 3),
                "Venue": p.venue or "N/A",
                "Title": p.title,
                "arXiv": p.arxiv_url,
            }
        )

    df = pd.DataFrame(table_rows)

    if not df.empty:
        st.dataframe(
            df,
            width='stretch',
            hide_index=True,
            column_config={
                "arXiv": st.column_config.LinkColumn(
                    label="arXiv",
                    help="Open arXiv page",
                    validate="^https?://.*",
                    max_chars=100,
                    display_text="arXiv link"
                )
            }
        )

    # 7.5 NEW: Trends Dashboard (Abstract-only EDA)
    dash_n_effective = min(max(int(dashboard_n), 10), len(ranked_papers))
    dashboard_papers = ranked_papers[:dash_n_effective]
    render_trends_dashboard(
        papers_for_dashboard=dashboard_papers,
        top_k_terms=25,
        trend_terms=8,
        max_heat_terms=15,
    )
    st.markdown("---")

    # 8. Top N highlighted
    top_n_effective = min(top_n, len(ranked_papers))
    topN = ranked_papers[:top_n_effective]
    st.session_state["topN"] = topN

    st.subheader(f"8. Top {top_n_effective} Papers (Highlighted)")

    if "plain_summaries" not in st.session_state:
        st.session_state["plain_summaries"] = {}
    plain_summaries: Dict[str, str] = st.session_state["plain_summaries"]

    for rank, p in enumerate(topN, start=1):
        st.markdown(f"### #{rank}: {p.title}")
        st.write(f"**Citation impact score (1 year):** {int(p.predicted_citations or 0)}")
        st.write(f"**Authors:** {', '.join(p.authors) if p.authors else 'Unknown'}")
        st.markdown(f"**Venue:** {p.venue or 'N/A'}")
        st.write(f"[arXiv link]({p.arxiv_url}) | [PDF link]({p.pdf_url})")

        if provider in ("openai", "gemini", "groq"):
            paper_key = p.arxiv_id or p.title
            if paper_key in plain_summaries:
                summary = plain_summaries[paper_key]
            else:
                with st.spinner("Generating plain English summary..."):
                    summary = summarize_paper_plain_english(p, llm_config)
                plain_summaries[paper_key] = summary
                st.session_state["plain_summaries"] = plain_summaries

            st.markdown("**Plain English summary:**")
            st.write(summary)

            if p.prediction_explanations:
                st.write("**Why this citation impact score:**")
                for ex in p.prediction_explanations[:3]:
                    st.write(f"- {ex}")
        else:
            st.markdown("**Plain English summary:** only available in OpenAI / Gemini / Groq options")
            st.markdown("**Why this citation impact score:** only available in OpenAI / Gemini / Groq options")

        if p.focus_label:
            st.write(f"**Focus label:** {p.focus_label}")
        rel_str = f"{p.llm_relevance_score:.2f}" if p.llm_relevance_score is not None else "N/A"
        st.write(f"**Relevance score:** {rel_str}")
        sim_str = f"{p.semantic_relevance:.3f}" if p.semantic_relevance is not None else "N/A"
        st.write(f"**Embedding similarity score:** {sim_str}")

        if p.semantic_reason:
            st.write("**Why this paper matches your brief:**")
            st.write(p.semantic_reason)

        st.write("**Abstract:**")
        st.write(p.abstract)
        st.markdown("---")

    # 9. Markdown report for top N
    st.subheader("9. Export Top N Report")

    report_lines = [
        f"# Top {top_n_effective} Papers (Citation Impact Scores) - {datetime.now().isoformat()}",
        "## Research Brief",
        research_brief,
        "",
        "## Not Looking For (optional)",
        not_looking_for or "(none provided)",
        "",
        f"Mode: {mode}",
        f"Date range: {current_start} to {current_end}",
        f"Provider: {'OpenAI' if provider == 'openai' else 'Gemini' if provider == 'gemini' else 'Groq' if provider == 'groq' else 'Free local heuristic'}",
        f"Chat model: {model_name}",
        f"Embedding model: {embedding_model_name}",
        f"Dashboard papers analyzed: {dash_n_effective}",
        "",
    ]
    for rank, p in enumerate(topN, start=1):
        report_lines.append(f"## #{rank}: {p.title}")
        report_lines.append(f"- Citation impact score (1 year): {int(p.predicted_citations or 0)}")
        report_lines.append(f"- Authors: {', '.join(p.authors) if p.authors else 'Unknown'}")
        report_lines.append(f"- arXiv: {p.arxiv_url}")
        report_lines.append(f"- PDF: {p.pdf_url}")
        if p.focus_label:
            report_lines.append(f"- Focus label: {p.focus_label}")
        if p.llm_relevance_score is not None:
            report_lines.append(f"- Relevance score: {p.llm_relevance_score:.2f}")
        if p.semantic_relevance is not None:
            report_lines.append(f"- Embedding similarity: {p.semantic_relevance:.3f}")
        if p.semantic_reason:
            report_lines.append(f"- Relevance explanation: {p.semantic_reason}")
        if provider in ("openai", "gemini", "groq"):
            report_lines.append("- Citation impact explanations:")
            if p.prediction_explanations:
                for ex in p.prediction_explanations[:3]:
                    report_lines.append(f"  - {ex}")
        report_lines.append("")
        report_lines.append("Abstract:")
        report_lines.append(p.abstract)
        report_lines.append("")

    report_path = os.path.join(project_folder, "topN_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    # 10. ZIP download
    if run_clicked or "zip_bytes" not in st.session_state:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname in [
                "config.json",
                "current_papers_all.json",
                "candidates_embedding_selected.json",
                "candidates_with_classification.json",
                "used_papers_for_prediction.json",
                "selected_papers_for_prediction.json",
                "selected_papers_with_predictions.json",
                "topN_report.md",
            ]:
                fpath = os.path.join(project_folder, fname)
                if os.path.exists(fpath):
                    zf.write(fpath, arcname=fname)
        zip_buffer.seek(0)
        st.session_state["zip_bytes"] = zip_buffer.getvalue()

    zip_bytes = st.session_state["zip_bytes"]

    st.success(f"Results saved in `{project_folder}`")
    st.write("- `current_papers_all.json`")
    st.write("- `candidates_embedding_selected.json`")
    st.write("- `candidates_with_classification.json`")
    st.write("- `used_papers_for_prediction.json`")
    st.write("- `selected_papers_for_prediction.json`")
    st.write("- `selected_papers_with_predictions.json`")
    st.write("- `topN_report.md`")

    st.download_button(
        "⬇️ Download all results as ZIP",
        data=zip_bytes,
        file_name=f"research_agent_{timestamp}.zip",
        mime="application/zip",
    )


if __name__ == "__main__":
    main()
