#!/usr/bin/env python3
"""FastAPI backend for PDF chat."""
import json
import re
from pathlib import Path
from contextlib import asynccontextmanager

import faiss
import httpx
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastembed import TextEmbedding
from langdetect import detect, LangDetectException
from stop_words import get_stop_words

# Config
INDEX_DIR = Path(__file__).parent.parent / "index"
LLAMA_SERVER = "http://127.0.0.1:8080"
TOP_K = 8
KEYWORD_BOOST = 5.0  # Huge boost for keyword matches to outrank semantic-only matches

SOURCE_PRIORITY = {
    "cv": 1.0,
    "skills": 1.0,
    "projekte": 1.0,
    "default": 0.5
}


def get_source_boost(source: str) -> float:
    src = source.lower()
    if "cv" in src or "lebenslauf" in src:
        return SOURCE_PRIORITY["cv"]
    if "skill" in src:
        return SOURCE_PRIORITY["skills"]
    if "projekt" in src:
        return SOURCE_PRIORITY["projekte"]
    return SOURCE_PRIORITY["default"]

# Globals
index = None
metadata = []
embedder = None
http_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global index, metadata, embedder, http_client

    # Load FAISS index
    index_path = INDEX_DIR / "index.faiss"
    meta_path = INDEX_DIR / "metadata.jsonl"

    if not index_path.exists():
        raise RuntimeError(f"Index not found: {index_path}")

    index = faiss.read_index(str(index_path))
    with open(meta_path) as f:
        metadata = [json.loads(line) for line in f]

    # Load embedding model
    embedder = TextEmbedding("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # HTTP client with connection pooling
    http_client = httpx.AsyncClient(timeout=120.0)

    print(f"Loaded index with {index.ntotal} vectors")
    yield

    await http_client.aclose()


app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]


@app.get("/health")
def health():
    return {"status": "ok", "chunks": index.ntotal if index else 0}


def normalize_text(text: str) -> str:
    """Normalize text by removing dots, underscores, replacing dashes with spaces for fuzzy matching."""
    text = text.lower()
    text = re.sub(r'[._]', '', text)  # Remove dots and underscores
    text = re.sub(r'-', ' ', text)    # Replace dashes with spaces to split hyphenated words
    return text


# Query expansion: map query signals to document terms
QUERY_EXPANSIONS = {
    # Employment-related queries (English)
    'work': ['gmbh', 'bellissy', 'winlocal', 'futureship', 'iwm', 'professional experience', 'software developer'],
    'worked': ['gmbh', 'bellissy', 'winlocal', 'futureship', 'iwm', 'professional experience'],
    'job': ['gmbh', 'bellissy', 'winlocal', 'futureship', 'iwm', 'software developer'],
    'jobs': ['gmbh', 'bellissy', 'winlocal', 'futureship', 'iwm', 'software developer'],
    'employer': ['gmbh', 'bellissy', 'winlocal', 'futureship', 'iwm'],
    'employers': ['gmbh', 'bellissy', 'winlocal', 'futureship', 'iwm'],
    'company': ['gmbh', 'bellissy', 'winlocal', 'futureship', 'iwm'],
    'companies': ['gmbh', 'bellissy', 'winlocal', 'futureship', 'iwm'],
    'experience': ['professional experience', 'bellissy', 'winlocal', 'futureship'],
    'career': ['professional experience', 'bellissy', 'winlocal', 'futureship', 'iwm'],
    'history': ['professional experience', 'bellissy', 'winlocal', 'futureship', 'iwm'],
    # Employment-related queries (German)
    'gearbeitet': ['gmbh', 'bellissy', 'winlocal', 'futureship', 'iwm', 'professional experience', 'software developer'],
    'arbeit': ['gmbh', 'bellissy', 'winlocal', 'futureship', 'iwm', 'professional experience'],
    'arbeitet': ['gmbh', 'bellissy', 'winlocal', 'futureship', 'iwm', 'professional experience'],
    'arbeitgeber': ['gmbh', 'bellissy', 'winlocal', 'futureship', 'iwm'],
    'firma': ['gmbh', 'bellissy', 'winlocal', 'futureship', 'iwm'],
    'firmen': ['gmbh', 'bellissy', 'winlocal', 'futureship', 'iwm'],
    'unternehmen': ['gmbh', 'bellissy', 'winlocal', 'futureship', 'iwm'],
    'beruf': ['gmbh', 'bellissy', 'winlocal', 'futureship', 'iwm', 'software developer'],
    'berufserfahrung': ['gmbh', 'bellissy', 'winlocal', 'futureship', 'iwm', 'professional experience'],
    'erfahrung': ['professional experience', 'bellissy', 'winlocal', 'futureship'],
    'karriere': ['professional experience', 'bellissy', 'winlocal', 'futureship', 'iwm'],
    'laufbahn': ['professional experience', 'bellissy', 'winlocal', 'futureship', 'iwm'],
    # Education-related queries (English)
    'study': ['university', 'koblenz', 'diploma', 'education', 'degree'],
    'studied': ['university', 'koblenz', 'diploma', 'education', 'degree'],
    'education': ['university', 'koblenz', 'diploma', 'degree', 'courses'],
    'degree': ['diploma', 'university', 'koblenz', 'computer science'],
    'university': ['koblenz', 'diploma', 'education', 'degree'],
    'courses': ['workshop', 'training', 'scrum', 'adwords'],
    # Education-related queries (German)
    'studiert': ['universität', 'koblenz', 'diplom', 'bildung'],
    'studium': ['universität', 'koblenz', 'diplom', 'bildung'],
    'ausbildung': ['universität', 'koblenz', 'diplom', 'education'],
    'abschluss': ['diplom', 'universität', 'koblenz'],
    # General/vague queries - expand to key document terms
    'background': ['professional experience', 'skills', 'bellissy', 'winlocal', 'software developer'],
    'summary': ['professional experience', 'skills', 'bellissy', 'winlocal', 'software developer'],
    'profile': ['professional experience', 'skills', 'software developer', 'backend'],
    'overview': ['professional experience', 'skills', 'bellissy', 'winlocal', 'software developer'],
    'hintergrund': ['professional experience', 'skills', 'bellissy', 'winlocal', 'software developer'],
    'zusammenfassung': ['professional experience', 'skills', 'bellissy', 'winlocal'],
    'profil': ['professional experience', 'skills', 'software developer', 'backend'],
    # Skills/technology queries - boost SKILLS.pdf
    'skill': ['skills', 'extended technical skills', 'core stack'],
    'skills': ['skills', 'extended technical skills', 'core stack'],
    'erfahrung': ['skills', 'extended technical skills', 'professional experience'],
    'kennt': ['skills', 'extended technical skills', 'core stack'],
    'kenntnisse': ['skills', 'extended technical skills', 'core stack'],
    'technologie': ['skills', 'extended technical skills', 'core stack'],
    'technologien': ['skills', 'extended technical skills', 'core stack'],
    'tech': ['skills', 'extended technical skills', 'core stack'],
    'stack': ['skills', 'extended technical skills', 'core stack'],
    'tools': ['skills', 'extended technical skills', 'core stack'],
    'framework': ['skills', 'extended technical skills', 'core stack'],
    'frameworks': ['skills', 'extended technical skills', 'core stack'],
    'devops': ['skills', 'docker', 'jenkins', 'ansible', 'deployment'],
    'infrastructure': ['skills', 'docker', 'hetzner', 'deployment'],
    'infrastruktur': ['skills', 'docker', 'hetzner', 'deployment'],
}


def expand_query(text: str) -> set[str]:
    """Expand query with related terms from document vocabulary."""
    text_lower = text.lower()
    expanded = set()
    for trigger, expansions in QUERY_EXPANSIONS.items():
        if trigger in text_lower:
            expanded.update(expansions)

    # Fallback for vague queries about the person with no specific triggers
    person_refs = ['conrad', 'emde', 'him', 'his', 'person', 'candidate', 'er', 'ihn', 'sein']
    has_person_ref = any(ref in text_lower for ref in person_refs)
    vague_verbs = ['tell', 'about', 'describe', 'who is', 'erzähl', 'über', 'wer ist', 'beschreib']
    has_vague_verb = any(v in text_lower for v in vague_verbs)

    if has_person_ref and has_vague_verb and not expanded:
        # Very vague query - add general expansion
        expanded.update(['professional experience', 'skills', 'bellissy', 'winlocal', 'software developer', 'backend'])

    return expanded


def is_skill_query(text: str) -> bool:
    """Detect if query is asking about skills/technologies."""
    text_lower = text.lower()
    # Patterns that indicate skill questions (DE + EN)
    skill_patterns = [
        'erfahrung', 'experience', 'worked with',
        'kennt', 'knows', 'know ',
        'verwendet', 'used', 'using',
        'gearbeitet', 'worked',
        'kann ', 'can ', 'able to',
        'beherrscht', 'proficient',
        'skill', 'technologie', 'technology',
        'tool', 'framework', 'library',
    ]
    if any(p in text_lower for p in skill_patterns):
        return True

    # Also check for known tech names / single word tech queries
    tech_names = {'kubernetes', 'docker', 'git', 'ansible', 'nodejs', 'python', 'php', 'javascript', 'typescript', 'react', 'vue', 'svelte', 'graphql', 'rest', 'sql', 'mysql', 'postgres', 'mongodb', 'redis', 'elasticsearch', 'nginx', 'apache', 'jenkins', 'gitlab', 'kubernetes'}
    words = set(text_lower.replace('-', ' ').replace('?', '').split())
    return bool(words & tech_names)


def is_german_query(text: str) -> bool:
    """Detect if query is in German using langdetect with fallback."""
    try:
        # Only try langdetect if text is long enough
        if len(text) < 5:
            # For very short queries, check for German characters or patterns
            german_indicators = {'ä', 'ö', 'ü', 'ß', '?'}
            if any(c in text.lower() for c in german_indicators):
                return True
            return False

        lang = detect(text)
        return lang == 'de'
    except LangDetectException:
        # Fallback: if detection fails, assume English
        return False


def extract_keywords(text: str, is_german: bool = None) -> set[str]:
    """Extract meaningful words from query, normalized for fuzzy matching."""
    normalized = normalize_text(text)
    words = re.findall(r'\b[a-zA-Z]{3,}\b', normalized)

    # Use standard stopwords + domain-specific
    if is_german is None:
        try:
            is_german = detect(text) == 'de'
        except:
            is_german = False

    lang = 'german' if is_german else 'english'
    std_stopwords = set(get_stop_words(lang))

    # Domain-specific stopwords (person name, generic terms)
    domain_stopwords = {'conrad', 'emde', 'info', 'information', 'details', 'provide'}

    return {w for w in words if w not in std_stopwords and w not in domain_stopwords}


async def rewrite_question(question: str, is_german: bool) -> str:
    """Rewrite vague questions to be more specific for better retrieval."""
    # For very short/vague questions, ask LLM to clarify
    if is_german:
        rewrite_prompt = f"""Reformuliere diese Frage präziser. Ersetze Pronomen mit konkreten Begriffen. Nur die reformulierte Frage zurückgeben, nichts anderes. Beziehe die Frage auf Conrad Emde.

        Beispiele:
        <example>
        Original: "Firmen?"
        Reformuliert: "Für welche Firmen hat Conrad Emde gearbeitet?"
        </example>
        <example>
        Original: "seine Fähigkeiten?"
        Reformuliert: "Welche Fähigkeiten hat Conrad Emde?"
        </example>
        <example>
        Original: "docker?"
        Reformuliert: "Welche beruflichen Erfahrungen hat Conrad Emde mit Docker?"
        </example>

Frage: {question}
Reformuliert:"""
        attach = " Gib eine ausführliche Antwort."
    else:
        rewrite_prompt = f"""Rewrite this question to be more specific. Replace pronouns with concrete terms. Return only the rewritten question, nothing else. Make the question about Conrad Emde.

        Examples:
        <example>
        Original: "companies?"
        Rewritten: "What companies has Conrad Emde worked for?"
        </example>
        <example>
        Original: "his skills?"
        Rewritten: "What skills does Conrad Emde have?"
        </example>
        <example>
        Original: "docker?"
        Rewritten: "What is Conrad Emde's professional experience with docker?"
        </example>
Question: {question}
Rewritten:"""
        attach = " Give a  detailed answer."

    try:
        resp = await http_client.post(
            f"{LLAMA_SERVER}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": rewrite_prompt}],
                "temperature": 0.1,
                "max_tokens": 100,
                "repeat_penalty": 1.0
            }
        )
        resp.raise_for_status()
        data = resp.json()
        rewritten = data["choices"][0]["message"]["content"].strip()
        print(f"Rewrote question: '{question}' -> '{rewritten + attach}'")
        # Only use rewrite if it's different and not empty
        if rewritten and rewritten != question and len(rewritten) < 200:
            return rewritten + attach
    except Exception:
        pass  # Fall back to original question if rewriting fails

    return question


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty")

    # Detect language and query type early for source boosting
    is_german = is_german_query(req.question)
    is_skill = is_skill_query(req.question)

    # For very short queries, assume German if it looks like a German tech question
    if not is_german and is_skill and req.question.endswith('?') and len(req.question) < 30:
        # Single tech word followed by ? - likely German from DE docs
        is_german = True

    # Rewrite vague questions for better retrieval
    search_question = await rewrite_question(req.question, is_german)

    # Extract keywords for boosting + query expansion
    keywords = extract_keywords(search_question, is_german)
    expanded = expand_query(search_question)
    keywords = keywords | expanded

    # For skill queries, also add the raw query words (to catch tech names like "Kubernetes")
    if is_skill:
        raw_words = set(re.findall(r'\b[a-zA-Z]{2,}\b', req.question.lower()))
        # Add all words except common question words
        question_words = {'hat', 'ist', 'mit', 'dem', 'der', 'die', 'das', 'ein', 'eine', 'haben', 'gibt', 'gibt', 'welche', 'welcher', 'was', 'wie', 'erfahrung', 'experience', 'kennt', 'knows', 'verwendet', 'used'}
        keywords = keywords | (raw_words - question_words)

    # Embed query
    query_vec = list(embedder.embed([req.question]))
    query_vec = np.array(query_vec, dtype=np.float32)
    faiss.normalize_L2(query_vec)

    # Search all chunks for reranking (small index)
    scores, indices = index.search(query_vec, index.ntotal)

    # Score with keyword boost
    candidates = []
    for i, idx in enumerate(indices[0]):
        if idx >= 0:
            chunk = metadata[idx]
            text_lower = chunk["text"].lower()

            # Keyword boost: add score for each keyword found
            text_normalized = normalize_text(chunk["text"])
            matched_kws = [kw for kw in keywords if kw in text_normalized]
            keyword_score = len(matched_kws) * KEYWORD_BOOST
            source_boost = get_source_boost(chunk["source"])

            # Boost German docs for German queries
            if is_german and 'projekt' in chunk["source"].lower():
                source_boost += 2.0

            # Boost SKILLS.pdf for skill/technology queries
            if is_skill and 'skill' in chunk["source"].lower():
                source_boost += 3.0

            combined_score = float(scores[0][i]) + keyword_score + source_boost

            candidates.append((combined_score, chunk))

    # Sort by combined score and take top K
    candidates.sort(key=lambda x: x[0], reverse=True)
    candidates = candidates[:TOP_K]

    # Reorder: put chunks with keyword matches first (helps small LLMs)
    def has_keyword(chunk):
        text_normalized = normalize_text(chunk["text"])
        return any(kw in text_normalized for kw in keywords)

    with_keywords = [(s, c) for s, c in candidates if has_keyword(c)]
    without_keywords = [(s, c) for s, c in candidates if not has_keyword(c)]
    candidates = with_keywords + without_keywords

    contexts = []
    sources = []
    for i, (score, chunk) in enumerate(candidates):
        contexts.append(f"[Segment {i+1}] Source: {chunk['source']}\n{chunk['text']}")
        sources.append({
            "source": chunk["source"],
            "chunk_id": chunk["chunk_id"],
            "score": score
        })

    if not contexts:
        return ChatResponse(answer="No relevant context found.", sources=[])

    # Check if top result is relevant
    # With keyword match: score = semantic (~0.3) + keyword_boost (5.0) + source (1.0) = ~6.3
    # Without keyword match: score = semantic (~0.3) + source (1.0) = ~1.3
    MIN_RELEVANCE_SCORE = 1.5
    if candidates[0][0] < MIN_RELEVANCE_SCORE:
        # For skill queries, try German response if looks like German question
        no_info_msg = "I don't have information about that in the provided documents."
        if is_skill and any(c in req.question.lower() for c in {'ä', 'ö', 'ü', 'ß', '?'}):
            no_info_msg = "Diese Information ist in den Dokumenten nicht enthalten."
        return ChatResponse(
            answer=no_info_msg,
            sources=[]
        )

    # Build prompts
    system_parts = [
        "You are an assistant answering questions about a candidate's professional background using the provided document segments.",
        "",
        "RULES:",
        "- Answer ONLY using information from the provided segments.",
        "- List ALL relevant items found across ALL segments when asked for lists.",
        "- When asked about work/employers/jobs, the ONLY employers are: Bellissy GmbH, WinLocal GmbH, FutureShip GmbH, IWM Koblenz.",
        "- IMPORTANT: Using a tool/API (like Google AdWords) does NOT mean working AT that company. Only list actual employers.",
        "- Do NOT mix information from different projects or companies.",
        "- Be comprehensive but concise.",
        "- If the information is not in the segments, say so.",
    ]

    if is_german:
        system_parts.extend([
            "",
            "DEUTSCHE ANFRAGE - Antworte AUF DEUTSCH:",
            "- Antworte IMMER auf Deutsch, niemals auf Englisch.",
            "- Gib vollständige, detaillierte Antworten - nicht nur Stichpunkte.",
            "- Bei 'keine Information': Sage 'Diese Information ist in den Dokumenten nicht enthalten.'",
            "- Liste ALLE relevanten Arbeitgeber auf: Bellissy GmbH, WinLocal GmbH, FutureShip GmbH, IWM Koblenz.",
            "- Nutze auch Inhalte aus deutschen Dokumenten (PROJEKTE.pdf).",
        ])
    
    context_str = "\n\n---\n\n".join(contexts)
    
    user_parts = [
        "CONTEXT:",
        context_str,
        "",
        "QUESTION: " + req.question,
        "",
        "ANSWER:"
    ]
    
    system_prompt = "\n".join(system_parts)
    user_prompt = system_prompt + "\n".join(user_parts)

    # Call llama-server
    try:
        resp = await http_client.post(
            f"{LLAMA_SERVER}/v1/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.2 if is_german else 0.0,  # Slightly more natural for German
                "max_tokens": 1024,   # German text tends to be longer
                "repeat_penalty": 1.1
            }
        )
        resp.raise_for_status()
        data = resp.json()
        answer = data["choices"][0]["message"]["content"]
    except httpx.HTTPStatusError as e:
        print(f"LLM Error Body: {e.response.text}")
        raise HTTPException(503, f"LLM server error: {e}")
    except httpx.RequestError as e:
        raise HTTPException(503, f"LLM server error: {e}")

    return ChatResponse(answer=answer, sources=sources)


# Mount static files AFTER routes
app.mount("/", StaticFiles(directory=Path(__file__).parent.parent / "web", html=True), name="static")
