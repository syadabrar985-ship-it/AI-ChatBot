```python
import os
import re
import time
from urllib.parse import urljoin, urlparse

import faiss
import numpy as np
import requests
import streamlit as st
from bs4 import BeautifulSoup
from groq import Groq
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

UNIVERSITY_NAME = "University of Baltistan, Skardu"

START_URLS = [
    "https://www.uobs.edu.pk/",
    "https://www.uobs.edu.pk/about/",
    "https://www.uobs.edu.pk/admissions/",
    "https://www.uobs.edu.pk/academics/faculties/",
    "https://www.uobs.edu.pk/general-pages/contact-us.php",
    "https://admissions.uobs.edu.pk/",
]

ALLOWED_DOMAINS = {
    "www.uobs.edu.pk",
    "uobs.edu.pk",
    "admissions.uobs.edu.pk",
}

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# This is a text-generation model.
GROQ_MODEL = "llama-3.3-70b-versatile"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 5

REQUEST_TIMEOUT = 20
MAX_PAGES = 80


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="UoBS AI Assistant",
    page_icon="🎓",
    layout="wide",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #6b7280;
        margin-bottom: 25px;
    }

    .info-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #f5f7fa;
        border: 1px solid #e5e7eb;
        margin-bottom: 15px;
    }

    .source-box {
        padding: 10px;
        border-radius: 8px;
        background-color: #f8fafc;
        border-left: 4px solid #2563eb;
        margin-top: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🎓 UoBS AI Assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "University of Baltistan, Skardu — AI-powered University Information Assistant"
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# API KEY
# ============================================================

def get_groq_api_key():
    """
    Reads GROQ_API_KEY from Streamlit secrets first,
    then from environment variables.
    """

    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

    return os.getenv("GROQ_API_KEY")


GROQ_API_KEY = get_groq_api_key()


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "index_data" not in st.session_state:
    st.session_state.index_data = None


# ============================================================
# WEB CRAWLER
# ============================================================

def clean_text(text):
    """
    Removes unnecessary whitespace from extracted webpage text.
    """

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_allowed_url(url):
    """
    Only crawl official UoBS domains.
    """

    parsed = urlparse(url)

    if parsed.scheme not in {"http", "https"}:
        return False

    return parsed.netloc.lower() in ALLOWED_DOMAINS


def normalize_url(url):
    """
    Removes URL fragments and normalizes trailing slashes.
    """

    parsed = urlparse(url)

    clean = parsed._replace(fragment="").geturl()

    if clean.endswith("/") and parsed.path != "/":
        clean = clean[:-1]

    return clean


def extract_page(url):
    """
    Downloads and extracts readable text from one webpage.
    """

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(compatible; UoBS-RAG-Assistant/1.0; "
            "+https://www.uobs.edu.pk/)"
        )
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(
        [
            "script",
            "style",
            "noscript",
            "svg",
            "header",
            "footer",
            "nav",
        ]
    ):
        tag.decompose()

    main_content = soup.find("main")

    if main_content:
        text = main_content.get_text(" ", strip=True)
    else:
        text = soup.get_text(" ", strip=True)

    text = clean_text(text)

    links = []

    for link in soup.find_all("a", href=True):
        absolute_url = normalize_url(
            urljoin(url, link["href"])
        )

        if is_allowed_url(absolute_url):
            links.append(absolute_url)

    return text, links


def crawl_university():
    """
    Crawls official UoBS pages and collects their text.
    """

    queue = list(START_URLS)
    visited = set()
    documents = []

    while queue and len(visited) < MAX_PAGES:

        current_url = normalize_url(queue.pop(0))

        if current_url in visited:
            continue

        visited.add(current_url)

        try:
            text, links = extract_page(current_url)

            if len(text) >= 100:
                documents.append(
                    {
                        "url": current_url,
                        "text": text,
                    }
                )

            for link in links:

                if (
                    link not in visited
                    and link not in queue
                    and len(queue) < MAX_PAGES * 2
                ):
                    queue.append(link)

        except requests.RequestException:
            continue

        except Exception:
            continue

    return documents


# ============================================================
# TEXT CHUNKING
# ============================================================

def create_chunks(documents):
    """
    Splits webpages into overlapping text chunks.
    """

    chunks = []

    for document in documents:

        text = document["text"]
        url = document["url"]

        start = 0

        while start < len(text):

            end = start + CHUNK_SIZE

            chunk_text = text[start:end]

            if len(chunk_text.strip()) >= 100:

                chunks.append(
                    {
                        "text": chunk_text.strip(),
                        "url": url,
                    }
                )

            if end >= len(text):
                break

            start = end - CHUNK_OVERLAP

    return chunks


# ============================================================
# EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():
    """
    Loads the sentence-transformers embedding model once.
    """

    return SentenceTransformer(EMBEDDING_MODEL)


# ============================================================
# VECTOR DATABASE
# ============================================================

def create_vector_database(chunks, model):
    """
    Creates embeddings and stores them inside FAISS.
    """

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32",
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index, chunks


# ============================================================
# BUILD RAG KNOWLEDGE BASE
# ============================================================

@st.cache_resource(show_spinner=False)
def build_knowledge_base():
    """
    Complete RAG ingestion pipeline:

    Website
       ↓
    Crawling
       ↓
    Text extraction
       ↓
    Chunking
       ↓
    Embeddings
       ↓
    FAISS vector database
    """

    documents = crawl_university()

    if not documents:
        raise RuntimeError(
            "No university webpages could be collected."
        )

    chunks = create_chunks(documents)

    if not chunks:
        raise RuntimeError(
            "No text chunks could be created."
        )

    embedding_model = load_embedding_model()

    index, chunks = create_vector_database(
        chunks,
        embedding_model,
    )

    return {
        "index": index,
        "chunks": chunks,
        "pages": documents,
    }


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_context(question, knowledge_base, top_k=TOP_K):
    """
    Converts the question into an embedding and retrieves
    the most relevant university chunks from FAISS.
    """

    model = load_embedding_model()

    question_embedding = model.encode(
        [question],
        normalize_embeddings=True,
    )

    question_embedding = np.asarray(
        question_embedding,
        dtype="float32",
    )

    scores, indices = knowledge_base["index"].search(
        question_embedding,
        top_k,
    )

    results = []

    for score, index_number in zip(
        scores[0],
        indices[0],
    ):

        if index_number == -1:
            continue

        chunk = knowledge_base["chunks"][index_number]

        results.append(
            {
                "text": chunk["text"],
                "url": chunk["url"],
                "score": float(score),
            }
        )

    return results


# ============================================================
# GROQ RESPONSE
# ============================================================

def generate_answer(question, retrieved_chunks):

    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is missing. "
            "Add it to Streamlit Secrets or your environment."
        )

    client = Groq(
        api_key=GROQ_API_KEY,
    )

    context_parts = []

    for number, item in enumerate(
        retrieved_chunks,
        start=1,
    ):
        context_parts.append(
            f"""
SOURCE {number}
URL: {item["url"]}

CONTENT:
{item["text"]}
"""
        )

    context = "\n".join(context_parts)

    system_prompt = f"""
You are the official-information assistant for
{UNIVERSITY_NAME}.

Your job is to answer questions about the university
using ONLY the supplied retrieved context.

Rules:

1. Do not invent university information.
2. Do not guess admission dates, fees, programs,
   faculty information, contact details, or policies.
3. If the answer is not contained in the context,
   clearly say that the information was not found
   in the current university knowledge base.
4. Give concise and useful answers.
5. When possible, mention the relevant official source URL.
6. If information appears time-sensitive, tell the user
   to verify the latest information on the official
   university website.
7. Never pretend that you accessed private student,
   faculty, or administrative accounts.
"""

    user_prompt = f"""
Retrieved University Context:

{context}

Student Question:

{question}

Answer the student's question using the retrieved
university information.
"""

    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0.1,
        max_tokens=800,
    )

    return completion.choices[0].message.content


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🎓 UoBS Assistant")

    st.write(
        "Ask questions about the University of "
        "Baltistan, Skardu."
    )

    st.divider()

    st.subheader("Knowledge Base")

    if st.session_state.index_data:

        pages = len(
            st.session_state.index_data["pages"]
        )

        chunks = len(
            st.session_state.index_data["chunks"]
        )

        st.success("Knowledge base ready")

        st.metric(
            "Web Pages",
            pages,
        )

        st.metric(
            "Text Chunks",
            chunks,
        )

    else:

        st.info(
            "The university website will be crawled "
            "when the knowledge base is initialized."
        )

    st.divider()

    if st.button(
        "🔄 Refresh University Data",
        use_container_width=True,
    ):

        st.cache_resource.clear()

        st.session_state.index_data = None

        st.session_state.messages = []

        st.rerun()

    st.divider()

    st.caption(
        "Data source: official University of "
        "Baltistan websites."
    )


# ============================================================
# INITIALIZE KNOWLEDGE BASE
# ============================================================

if st.session_state.index_data is None:

    with st.spinner(
        "Collecting and indexing University of Baltistan data..."
    ):

        try:

            st.session_state.index_data = (
                build_knowledge_base()
            )

        except Exception as error:

            st.error(
                "The university knowledge base could not "
                "be initialized."
            )

            st.exception(error)

            st.stop()


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if message.get("sources"):

            with st.expander(
                "📚 Sources"
            ):

                for source in message["sources"]:

                    st.markdown(
                        f"- [{source}]({source})"
                    )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask anything about University of Baltistan..."
)


if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    try:

        with st.chat_message("assistant"):

            with st.spinner(
                "Searching university knowledge..."
            ):

                retrieved = retrieve_context(
                    question,
                    st.session_state.index_data,
                )

            if not retrieved:

                answer = (
                    "I could not find relevant information "
                    "in the current University of Baltistan "
                    "knowledge base."
                )

                sources = []

            else:

                answer = generate_answer(
                    question,
                    retrieved,
                )

                sources = list(
                    dict.fromkeys(
                        item["url"]
                        for item in retrieved
                    )
                )

            st.markdown(answer)

            if sources:

                with st.expander(
                    "📚 Official Sources"
                ):

                    for source in sources:

                        st.markdown(
                            f"- [{source}]({source})"
                        )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                }
            )

    except Exception as error:

        error_message = (
            "Something went wrong while processing "
            "your question."
        )

        st.error(error_message)

        with st.expander(
            "Technical details"
        ):
            st.exception(error)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": error_message,
                "sources": [],
            }
        )
```
