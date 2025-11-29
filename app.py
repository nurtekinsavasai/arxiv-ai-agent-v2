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

try:
    import streamlit as st
    import requests
    import feedparser
    import pandas as pd
    from openai import OpenAI, NotFoundError, BadRequestError
except ImportError as e:
    missing = str(e).split("'")[1]
    print(f"Missing package: {missing}")
    print("Please run: pip install streamlit requests feedparser openai pandas")
    raise

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


# =========================
# Data structures
# =========================

@dataclass
class LLMConfig:
    api_key: str
    model: str
    api_base: str
    provider: str = "openai"  # "openai", "gemini", or "free_local"


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
    """
    Split the NOT field into simple terms or phrases.
    Supports comma, semicolon, or newline separated entries.
    """
    not_text = not_text.strip()
    if not not_text:
        return []
    parts = re.split(r"[,\n;]+", not_text)
    terms = [p.strip().lower() for p in parts if p.strip()]
    return terms


def filter_papers_by_not_terms(papers: List[Paper], not_text: str) -> (List[Paper], int):
    """
    Remove any paper whose title or abstract contains at least one NOT term
    (case insensitive).
    Returns the filtered list and the number of removed papers.
    """
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

        # All conferences
        if venue_filter_type == "All Conferences":
            if any(conf.lower() in venue for conf in CONFERENCE_KEYWORDS):
                filtered.append(p)

        # All journals
        elif venue_filter_type == "All Journals":
            if any(j.lower() in venue for j in JOURNAL_KEYWORDS):
                filtered.append(p)

        # Specific venue(s) in a chosen category
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

# Updated list (use NeurIPS only)
CONFERENCE_KEYWORDS = [
    "EMNLP", "ACL", "NAACL", "EACL",
    "NeurIPS", "ICLR", "ICML",
    "CVPR", "ECCV",
    "ICASSP", "AAAI", "AISTATS",
]

JOURNAL_KEYWORDS = [
    "IEEE Transactions", "ACM Transactions", "Transactions on",
    "Nature", "Science", "Elsevier", "Springer", "Journal",
]

NEGATIVE_VENUE_SIGNALS = ["submitted to", "under review", "preprint"]


def extract_venue(comment: str) -> Optional[str]:
    """Extract conference or journal info from arXiv comments, ignoring submissions."""
    if not comment:
        return None

    c = comment.lower()

    # ❌ Ignore papers that are only 'submitted', 'under review', or 'preprint'
    if any(sig in c for sig in NEGATIVE_VENUE_SIGNALS):
        return None

    # Combine and sort for deterministic matching
    all_venues = sorted(CONFERENCE_KEYWORDS + JOURNAL_KEYWORDS)

    for venue in all_venues:
        if venue.lower() in c:
            return venue

    return None


# =========================
# Robust arXiv fetching
# =========================

def fetch_arxiv_papers_by_date(
    start_date: date,
    end_date: date,
    batch_size: int = 50,
    max_batches: int = 100,
    max_retries: int = 3,
) -> List[Paper]:
    """
    Fetch cs.AI + cs.LG + cs.HC papers from arXiv between start_date and end_date (inclusive).
    """
    query = "(cat:cs.AI OR cat:cs.LG OR cat:cs.HC)"
    base_url = "https://export.arxiv.org/api/query"

    results: List[Paper] = []
    seen_ids = set()  # Track arXiv IDs to avoid duplicates across batches
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
                st.error(
                    "Network error while calling arXiv. This might be a temporary issue on arxiv.org. "
                    "Try running again in a few seconds.\n\n"
                    f"Details: {e}"
                )
                return results

            if response.status_code == 429:
                retries += 1
                if retries > max_retries:
                    st.error(
                        "arXiv returned HTTP 429 (rate limit) repeatedly. "
                        f"Collected {len(results)} papers so far. "
                        "Try a shorter date range or run again later."
                    )
                    return results
                wait_seconds = 5 * (2 ** (retries - 1))
                st.warning(
                    f"arXiv rate limit (HTTP 429) encountered. "
                    f"Waiting {wait_seconds} seconds before retry {retries}/{max_retries}."
                )
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

            if published_date < start_date:
                return results

            # Extract arXiv ID early and deduplicate across all batches
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

            # Extract arXiv comment (contains venue info)
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
        time.sleep(1.0)

    if len(results) >= max_batches * batch_size:
        st.warning(
            f"Reached safety limit of {max_batches * batch_size} papers from arXiv. "
            "Consider using a shorter date range if you need more precise coverage."
        )

    return results


# =========================
# Generic LLM call + JSON helper (OpenAI / Gemini)
# =========================

def call_llm(prompt: str, llm_config: LLMConfig, label: str = "") -> str:
    if "last_prompts" not in st.session_state:
        st.session_state["last_prompts"] = {}
    st.session_state["last_prompts"][label or "default"] = prompt

    try:
        if llm_config.provider == "openai":
            client = OpenAI(
                api_key=llm_config.api_key,
                base_url=llm_config.api_base,
            )
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt},
            ]
            kwargs: Dict[str, Any] = {
                "model": llm_config.model,
                "messages": messages,
            }
            if not llm_config.model.startswith("o1"):
                kwargs["temperature"] = 0.2

            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content

        elif llm_config.provider == "gemini":
            if genai is None:
                st.error(
                    "Gemini provider selected but the google-genai package is not installed.\n\n"
                    "Run `pip install google-genai` in your environment."
                )
                st.stop()

            client = genai.Client(api_key=llm_config.api_key)
            response = client.models.generate_content(
                model=llm_config.model,
                contents=prompt,
            )
            # google-genai responses have a .text convenience property
            return response.text

        else:
            raise ValueError(f"Unknown provider: {llm_config.provider}")

    except NotFoundError:
        st.error(
            f"The model `{llm_config.model}` is not available for your API key. "
            "Please choose a different model, for example `gpt-4.1`, `gpt-4.1-mini`, "
            "`gpt-4o`, or `gpt-4o-mini`."
        )
        st.stop()
    except BadRequestError as e:
        st.error(
            f"Bad request when calling the model `{llm_config.model}`. "
            f"Details: {e}"
        )
        st.stop()
    except Exception as e:
        st.error(f"LLM call failed ({label or 'general'}): {e}")
        st.stop()


def safe_parse_json_array(raw: str) -> Optional[List[Dict[str, Any]]]:
    if not raw or not raw.strip():
        return None

    text = raw.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
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
        except Exception:
            pass

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        return None

    return None


# =========================
# Embeddings
# =========================

def embed_texts_openai(
    texts: List[str],
    llm_config: LLMConfig,
    embedding_model: str,
) -> List[List[float]]:
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


def embed_texts_gemini(
    texts: List[str],
    llm_config: LLMConfig,
    embedding_model: str,
) -> List[List[float]]:
    if not texts:
        return []
    if genai is None:
        st.error(
            "Gemini provider selected but the google-genai package is not installed.\n\n"
            "Run `pip install google-genai` in your environment."
        )
        st.stop()

    try:
        client = genai.Client(api_key=llm_config.api_key)
    except Exception as e:
        st.error(f"Failed to initialize Gemini client: {e}")
        st.stop()

    all_embeddings: List[List[float]] = []
    batch_size = 100  # Gemini API limit is 100 per request

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        try:
            # embed_content can take multiple contents
            response = client.models.embed_content(
                model=embedding_model,
                contents=batch,
            )
        except Exception as e:
            st.error(f"Gemini embedding API call failed: {e}")
            raise

        # response.embeddings is a list of embeddings with .values
        for emb in getattr(response, "embeddings", []):
            all_embeddings.append(list(emb.values))

    return all_embeddings


@st.cache_resource(show_spinner=False)
def get_local_embed_model() -> SentenceTransformer:
    if SentenceTransformer is None:
        raise RuntimeError(
            "sentence-transformers is not installed. "
            "Run `pip install sentence-transformers` in your environment."
        )
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def embed_texts_local(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    try:
        model = get_local_embed_model()
    except Exception as e:
        st.error(f"Local embedding model is not available. Details: {e}")
        st.stop()
    vectors = model.encode(texts, convert_to_numpy=True)
    return vectors.tolist()


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    dot = 0.0
    norm1 = 0.0
    norm2 = 0.0
    for a, b in zip(vec1, vec2):
        dot += a * b
        norm1 += a * a
        norm2 += b * b
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
    """
    From all fetched papers, pick the top-K most semantically similar to the brief.
    provider: "openai", "gemini", or "free_local"
    """
    if not papers:
        return []

    if provider in ("openai", "gemini") and llm_config is None:
        st.error(
            f"Internal error: llm_config is required for provider '{provider}'. "
            "Falling back to all papers."
        )
        return papers

    # 1) Embed the query
    try:
        if provider == "openai":
            query_vec = embed_texts_openai(
                [query_brief],
                llm_config=llm_config,  # type: ignore
                embedding_model=embedding_model,
            )[0]
        elif provider == "gemini":
            query_vec = embed_texts_gemini(
                [query_brief],
                llm_config=llm_config,  # type: ignore
                embedding_model=embedding_model,
            )[0]
        else:
            query_vec = embed_texts_local([query_brief])[0]
    except Exception:
        return papers

    # 2) Embed all papers
    texts = [p.title + "\n\n" + p.abstract for p in papers]
    try:
        if provider == "openai":
            paper_vecs = embed_texts_openai(
                texts,
                llm_config=llm_config,  # type: ignore
                embedding_model=embedding_model,
            )
        elif provider == "gemini":
            paper_vecs = embed_texts_gemini(
                texts,
                llm_config=llm_config,  # type: ignore
                embedding_model=embedding_model,
            )
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
    candidates = [p for _, p in scored[:k]]
    return candidates


# =========================
# LLM relevance classification (OpenAI / Gemini) + heuristic classification (free)
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

        Research brief (what the user is looking for and possibly what they are NOT looking for):
        \"\"\"{query_brief}\"\"\"

        For each paper, decide:

          1. focus_label: one of "primary", "secondary", or "off-topic".

             - "primary": The paper's MAIN contribution clearly and directly addresses
               the research brief.
             - "secondary": The paper is mainly about something else and the user's
               topic only appears as a minor application or example.
             - "off-topic": The paper does not meaningfully address the research brief.

          2. relevance_score: a float between 0.0 and 1.0 for how well the paper serves
             the user's interests, given its focus_label.

          3. reason: a 1–2 sentence explanation for the label.

        Return a JSON array, one entry per paper:
          {{
            "index": <integer index of the paper as given>,
            "focus_label": "primary" | "secondary" | "off-topic",
            "relevance_score": <float between 0.0 and 1.0>,
            "reason": "<short explanation>"
          }}

        No extra commentary outside the JSON.
        """).strip()

        prompt = "\n\n".join([instruction, "PAPERS:", *paper_blocks])
        raw = call_llm(prompt, llm_config, label="classification")

        parsed = safe_parse_json_array(raw)
        if parsed is None:
            st.error(
                "Failed to parse LLM classification output as JSON for one batch. "
                "Marking this batch as secondary with neutral scores."
            )
            for p in batch:
                p.focus_label = "secondary"
                p.llm_relevance_score = 0.5
                if p.semantic_reason is None:
                    p.semantic_reason = "LLM classification failed. Defaulted to secondary relevance."
            continue

        idx_to_info: Dict[int, Dict[str, Any]] = {}
        for item in parsed:
            try:
                idx = int(item["index"])
                label = str(item.get("focus_label", "")).strip().lower()
                if label not in ["primary", "secondary", "off-topic"]:
                    label = "off-topic"
                score = float(item.get("relevance_score", 0.0))
                reason = str(item.get("reason", "")).strip()
                idx_to_info[idx] = {
                    "focus_label": label,
                    "relevance_score": score,
                    "reason": reason,
                }
            except Exception:
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
                if p.semantic_reason is None:
                    p.semantic_reason = "No classification information returned. Treated as off-topic."

    return papers


def heuristic_classify_papers_free(candidates: List[Paper]) -> List[Paper]:
    if not candidates:
        return candidates

    ranked = sorted(
        candidates,
        key=lambda p: p.semantic_relevance if p.semantic_relevance is not None else 0.0,
        reverse=True,
    )

    n = len(ranked)
    if n == 0:
        return ranked

    top_k = max(1, min(n, max(10, int(0.3 * n))))

    for idx, p in enumerate(ranked):
        sim = p.semantic_relevance if p.semantic_relevance is not None else 0.0
        p.llm_relevance_score = sim
        if idx < top_k:
            p.focus_label = "primary"
        else:
            p.focus_label = "secondary"
        if p.semantic_reason is None:
            p.semantic_reason = "Heuristic classification in free local mode based on embedding similarity."

    return ranked


# =========================
# Direct citation impact scoring (OpenAI / Gemini) + heuristic scores (free)
# =========================

def build_direct_prediction_prompt(target_papers: List[Paper]) -> str:
    paper_blocks = []
    for i, p in enumerate(target_papers, start=1):
        block = textwrap.dedent(f"""
        Paper {i}:
        Title: {p.title}
        Authors: {", ".join(p.authors) if p.authors else "Unknown"}
        Email domains: {", ".join(p.email_domains) if p.email_domains else "Unknown"}
        Abstract: {p.abstract}
        """).strip()
        paper_blocks.append(block)

    instruction = textwrap.dedent("""
    You are an expert in computer science and scientometrics.

    Below are recently published computer science papers.
    For each paper, estimate a 1-year citation impact score. Treat this as a relative impact measure
    rather than a precise forecast. Higher scores indicate papers that are more likely to be widely
    read and cited, but the ranking across papers matters more than the absolute counts.

    Base your impact scores on:
      - Topic popularity and trendiness
      - Novelty and depth of the abstract
      - Breadth of potential audience and applicability
      - Any hints of strong affiliations or well known authors

    Return a JSON array. Each element must be:
      {
        "title": "<exact title of the paper>",
        "predicted_citations": <integer citation impact score>,
        "explanations": [
          "<short explanation 1>",
          "<short explanation 2>",
          "<short explanation 3>"
        ]
      }

    No extra commentary outside the JSON.
    """)

    prompt = "\n\n".join([instruction, "PAPERS:", *paper_blocks])
    return prompt


def predict_citations_direct(
    target_papers: List[Paper],
    llm_config: LLMConfig,
    batch_size: int = 8,
) -> List[Paper]:
    if not target_papers:
        return target_papers

    title_to_paper: Dict[str, Paper] = {p.title: p for p in target_papers}

    for start in range(0, len(target_papers), batch_size):
        batch = target_papers[start:start + batch_size]
        prompt = build_direct_prediction_prompt(batch)
        llm_output = call_llm(prompt, llm_config, label="prediction_batch")

        parsed = safe_parse_json_array(llm_output)
        if parsed is None:
            st.error(
                "Failed to parse LLM output as JSON for one citation impact scoring batch. "
                "Showing a raw snippet below for debugging."
            )
            st.code(llm_output[:1000])
            continue

        for item in parsed:
            title = item.get("title")
            if not isinstance(title, str):
                continue
            p = title_to_paper.get(title)
            if not p:
                continue
            try:
                p.predicted_citations = float(item.get("predicted_citations", 0))
            except Exception:
                p.predicted_citations = 0.0

            explanations = item.get("explanations", [])
            if isinstance(explanations, list):
                p.prediction_explanations = [str(ex) for ex in explanations[:3]]

    return list(title_to_paper.values())


def assign_heuristic_citations_free(papers: List[Paper]) -> List[Paper]:
    if not papers:
        return papers

    scores: List[float] = []
    for p in papers:
        rel = p.llm_relevance_score if p.llm_relevance_score is not None else 0.0
        sem = p.semantic_relevance if p.semantic_relevance is not None else 0.0
        score = 0.7 * rel + 0.3 * sem
        scores.append(score)

    min_s = min(scores)
    max_s = max(scores)
    for p, s in zip(papers, scores):
        if max_s > min_s:
            norm = (s - min_s) / (max_s - min_s)
        else:
            norm = 0.5
        p.predicted_citations = float(int(10 + norm * 40))

    return papers


# =========================
# Plain English summary helper (LLM modes)
# =========================

def summarize_paper_plain_english(paper: Paper, llm_config: LLMConfig) -> str:
    prompt = textwrap.dedent(f"""
    You are explaining a research paper to a smart reader who is NOT a machine learning expert.

    Paper title:
    \"\"\"{paper.title}\"\"\"

    Abstract:
    \"\"\"{paper.abstract}\"\"\"

    Based ONLY on this abstract (do not invent extra sections or experiments you do not see),
    write a short, plain-English summary that covers:

    - What this paper is about in one or two sentences.
    - Why it matters or what problem it is trying to solve.
    - The main idea or approach in simple terms.
    - One or two key takeaways for a non-technical reader.

    Avoid heavy jargon, equations, or implementation details. Aim for 3–6 short bullet points or short paragraphs.
    """).strip()

    summary = call_llm(prompt, llm_config, label="plain_english_summary")
    return summary


# =========================
# Pipeline description text
# =========================

PIPELINE_DESCRIPTION_MD = """
#### Describe what you want

You write a short research brief in natural language about the kind of work you care about, and optionally what you are not interested in. If you leave both fields empty, the agent switches to a global mode and just looks for the most impactful recent cs.AI, cs.LG, and cs.HC papers overall.

#### The agent fetches recent arXiv papers

It fetches up to about 5000 papers from arxiv.org in the Artificial Intelligence, Machine Learning, and Human–Computer Interaction categories (`cs.AI`, `cs.LG`, and `cs.HC`) for the date range you choose.

#### The agent picks candidate papers

- In **targeted mode**, the agent uses embeddings to measure how close each paper's title and abstract are to your brief in meaning and keeps the top 150 as candidates.  
- In **global mode**, it simply takes the most recent 150 `cs.AI`, `cs.LG`, and `cs.HC` papers as candidates.

#### The agent judges how relevant each paper is

- In **LLM API mode** (OpenAI or Gemini), a model reads each candidate and labels it as primary, secondary, or off topic.  
- In **free local mode**, the agent uses a simple heuristic based on the embedding similarity to mark the most relevant papers as primary and the rest as secondary.

#### The agent builds a citation impact set

The agent builds a set of papers to send to the citation impact step:

- It keeps all **primary** papers.  
- If there are fewer than about 20, it tops up with the strongest **secondary** papers until it reaches roughly 20, when possible.  
- In global mode, all candidates are used.

#### The agent computes 1-year citation impact scores

- In **LLM API mode** (OpenAI or Gemini), a model estimates a 1-year citation impact score for each paper and provides short explanations.  
- In **free local mode**, the agent derives a citation impact score from the relevance signals and uses that to rank papers.

These scores are heuristic impact signals and are best used for ranking within this batch, not as ground truth.

#### The agent ranks, summarizes, and saves results

The agent ranks papers, always showing **primary** papers first, then secondary ones. For the top N that you choose, it shows metadata, relevance signals, and links to arXiv and the PDF. In LLM API mode it also adds plain English summaries. All artifacts and a markdown report are saved in a project folder under `~/arxiv_ai_digest_projects/project_<timestamp>`, and you can download everything as a ZIP.
"""


# =========================
# Streamlit UI
# =========================

def main():
    st.set_page_config(
        page_title="Research Agent",
        layout="wide",
    )

    st.title("🔎 Research Agent")

    top_col1, top_col2 = st.columns([3, 1])
    with top_col1:
        st.write(
            """Too many important papers get lost in the noise. Most researchers and practitioners cannot reliably scan what is new recently in their area, find truly promising work, and trust that they did not miss something big."""
            " This agent helps with this problem by finding, ranking, and explaining recent AI papers on arxiv.org."
        )
    with top_col2:
        st.link_button(
            "▶ Watch a short demo",
            "https://youtu.be/4CvYLwlhXac"
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

            # First dropdown: choose category
            selected_category = st.selectbox(
                "Select type:",
                ["Conference", "Journal"]
            )

            # Depending on category show MULTISELECT
            if selected_category == "Conference":
                options = sorted(CONFERENCE_KEYWORDS)
            else:
                options = sorted(JOURNAL_KEYWORDS)

            selected_venues = st.multiselect(
                f"Select {selected_category.lower()}(s):",
                options=options
            )






        date_option = st.selectbox("Date Range", ["Last 3 Days", "Last Week", "Last Month"])

        st.markdown("### ⭐ Top N Highlight")
        top_n = st.slider(
            "How many top papers to highlight?",
            1, 10, 3
        )

        st.markdown("### 🔌 Provider")

        provider_label_free = "Free local model (no API key)"
        provider_label_openai = "OpenAI (API key required)"
        provider_label_gemini = "Gemini (API key required)"

        provider_choice = st.radio(
            "Choose provider",
            [provider_label_free, provider_label_openai, provider_label_gemini],
            index=0,
        )

        if provider_choice == provider_label_openai:
            provider = "openai"
        elif provider_choice == provider_label_gemini:
            provider = "gemini"
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
                "gpt-4.1-mini",
                "gpt-4.1",
                "gpt-4o-mini",
                "gpt-4o",
                "o1",
                "Custom",
            ]
            model_choice = st.selectbox(
                "OpenAI Chat model (for classification & citation impact scoring)",
                openai_models,
                index=0,
            )
            if model_choice == "Custom":
                model_name = st.text_input(
                    "Custom OpenAI Chat model name",
                    value="gpt-4.1-mini",
                    help="Example: gpt-4.1, gpt-4.1-mini, gpt-4o, gpt-4o-mini, o1, etc."
                )
            else:
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

            # Updated Gemini models list including Gemini 3 Preview
            gemini_models = [
                "gemini-3-pro-preview",
                "gemini-2.5-flash",
                "gemini-2.5-pro",
                "gemini-2.0-flash-exp",
                "Custom",
            ]
            gemini_choice = st.selectbox(
                "Gemini Chat model (for classification & citation impact scoring)",
                gemini_models,
                index=0,
            )
            if gemini_choice == "Custom":
                model_name = st.text_input(
                    "Custom Gemini model name",
                    value="gemini-3-pro-preview",
                    help="Use the model identifier shown in Google AI Studio, for example `gemini-3-pro-preview`."
                )
            else:
                model_name = gemini_choice

            embedding_model_name = GEMINI_EMBEDDING_MODEL_NAME
            st.caption(f"Embeddings (Gemini): `{embedding_model_name}`")

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

    # Mode and query brief
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
        "model_name": model_name,
        "provider": provider,
        "venue_filter_type": venue_filter_type,
        "selected_category": selected_category,
        "selected_venues": selected_venues,
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
            else "FreeLocalHeuristic"
        ),
        "top_n": top_n,
        "min_for_prediction": MIN_FOR_PREDICTION,
    }
    st.session_state["config"] = config
    save_json(os.path.join(project_folder, "config.json"), config)

    # 2. Fetch current papers
    st.subheader("2. Fetch Current Papers from arXiv (cs.AI + cs.LG + cs.HC)")

    if run_clicked or "current_papers" not in st.session_state:
        with st.spinner("Fetching cs.AI, cs.LG, and cs.HC papers from arXiv by date window..."):
            current_papers = fetch_arxiv_papers_by_date(
                start_date=current_start,
                end_date=current_end,
            )
        st.session_state["current_papers"] = current_papers
    else:
        current_papers = st.session_state["current_papers"]

    if not current_papers:
        st.warning("No cs.AI, cs.LG, or cs.HC papers found for this date range (or arXiv stopped responding).")
        return

    st.success(
        f"Fetched {len(current_papers)} cs.AI, cs.LG, and cs.HC papers in this date range "
        "(before any candidate selection)."
    )

    # Apply NOT filter provider-agnostically
    # NOT filter
    if not_text:
        current_papers, removed_count = filter_papers_by_not_terms(current_papers, not_text)
        st.info(f"Excluded {removed_count} papers whose title or abstract contained NOT terms.")



    venue_filter_type = params["venue_filter_type"]
    selected_category = params.get("selected_category")
    selected_venues = params.get("selected_venues", [])

    before_count = len(current_papers)
    current_papers = filter_papers_by_venue(
        current_papers,
        venue_filter_type,
        selected_category,
        selected_venues
    )
    after_count = len(current_papers)

    if venue_filter_type != "None":
        st.info(
            f"Venue filter `{venue_filter_type}` applied → "
            f"{selected_category or ''}: {', '.join(selected_venues) if selected_venues else 'None'}\n"
            f"Remaining: {after_count} (Filtered out {before_count - after_count})"
        )



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
                    llm_config=llm_config if provider in ("openai", "gemini") else None,
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
        if provider in ("openai", "gemini"):
            provider_label = "OpenAI" if provider == "openai" else "Gemini"
            if run_clicked or any(p.focus_label is None for p in candidates):
                with st.spinner(f"Classifying candidates as PRIMARY, SECONDARY, or OFF TOPIC ({provider_label})..."):
                    candidates = classify_papers_with_llm(
                        candidates,
                        query_brief=query_brief,
                        llm_config=llm_config,
                        batch_size=15,
                    )
                st.session_state["candidates"] = candidates
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
                        (
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

    if provider == "openai":
        st.markdown("""
**How this step works (OpenAI mode)**

For each selected paper, the agent sends the title, authors, and abstract to an OpenAI model and asks it to assign a 1-year citation impact score. The model bases this score on signals such as how trendy the topic is, how novel and substantial the abstract sounds, how broad the potential audience is, and whether the work appears to come from strong labs or well known authors.

These citation impact scores are heuristic and are best used for ranking and prioritization within this batch of papers, not as ground truth. They may reflect existing academic biases.
        """)
    elif provider == "gemini":
        st.markdown("""
**How this step works (Gemini mode)**

For each selected paper, the agent sends the title, authors, and abstract to a Gemini model (for example `gemini-3-pro-preview`) and asks it to assign a 1-year citation impact score. The model bases this score on signals such as how trendy the topic is, how novel and substantial the abstract sounds, how broad the potential audience is, and whether the work appears to come from strong labs or well known authors.

These citation impact scores are heuristic and are best used for ranking within this batch of papers, not as ground truth. They may reflect existing academic and data biases.
        """)
    else:
        st.markdown("""
**How this step works (free local mode)**

In free local mode, the agent does not call any external LLM. Instead, it combines the embedding based similarity and relevance scores into a single numeric citation impact score and uses that score as a proxy for how influential the paper might be relative to others in this batch. The absolute numbers are less important than the relative ranking.

These scores are heuristic and should be used as a guide for exploration rather than as formal evaluation metrics.
        """)

    if run_clicked or "ranked_papers" not in st.session_state:
        if provider in ("openai", "gemini"):
            with st.spinner("Calling LLM API to compute citation impact scores for selected papers..."):
                papers_with_pred = predict_citations_direct(
                    target_papers=selected_papers,
                    llm_config=llm_config,
                )
        else:
            with st.spinner("Computing heuristic citation impact scores from relevance signals..."):
                papers_with_pred = assign_heuristic_citations_free(selected_papers)

        papers_with_pred = [
            p for p in papers_with_pred if p.predicted_citations is not None
        ]
        if not papers_with_pred:
            st.error("Citation impact scoring did not produce any scores.")
            return

        primary_pred = [p for p in papers_with_pred if p.focus_label == "primary"]
        secondary_pred = [p for p in papers_with_pred if p.focus_label == "secondary"]
        others_pred = [p for p in papers_with_pred if p.focus_label not in ("primary", "secondary")]

        def sort_by_pred(papers: List[Paper]) -> List[Paper]:
            return sorted(
                papers,
                key=lambda p: (
                    p.predicted_citations if p.predicted_citations is not None else 0,
                ),
                reverse=True,
            )

        primary_pred_sorted = sort_by_pred(primary_pred)
        secondary_pred_sorted = sort_by_pred(secondary_pred)
        others_pred_sorted = sort_by_pred(others_pred)

        ranked_papers = primary_pred_sorted + secondary_pred_sorted + others_pred_sorted
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

    st.caption(
        "Primary papers appear first, ranked by citation impact score, followed by secondary papers."
    )

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
                "Venue":p.venue or "N/A",
                "Title": p.title,
                "arXiv": p.arxiv_url,

            }
        )

    df = pd.DataFrame(table_rows)

    if not df.empty:
        st.dataframe(
            df,
            use_container_width=True,
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
        st.write(f"**Venue:**{p.venue or 'N/'}")
        st.write(f"[arXiv link]({p.arxiv_url}) | [PDF link]({p.pdf_url})")

        if provider in ("openai", "gemini"):
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
                st.write("**Why this citation impact score (3 factors):**")
                for ex in p.prediction_explanations[:3]:
                    st.write(f"- {ex}")
        else:
            st.markdown("**Plain English summary:** only available in OpenAI / Gemini options")
            st.markdown("**Why this citation impact score (3 factors):** only available in OpenAI / Gemini options")

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
        f"Provider: {'OpenAI' if provider == 'openai' else 'Gemini' if provider == 'gemini' else 'Free local heuristic'}",
        f"Chat model: {model_name}",
        f"Embedding model: {embedding_model_name}",
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
        if provider in ("openai", "gemini"):
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
