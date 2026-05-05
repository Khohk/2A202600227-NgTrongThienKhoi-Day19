"""
Flat RAG using ChromaDB for baseline comparison.
No dependency on langchain.chains — calls OpenAI directly.
"""

import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
DATA_DIR   = "data/raw"
_oai       = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL      = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

ANSWER_PROMPT = """Use ONLY the context below to answer the question.
If the context does not contain enough information, say "Not found in context."

Context:
{context}

Question: {question}
Answer:"""


def build_flat_index():
    loader = DirectoryLoader(DATA_DIR, glob="**/*.md",
                             loader_cls=TextLoader,
                             loader_kwargs={"encoding": "utf-8"})
    docs   = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DIR)
    print(f"Flat RAG index built: {len(chunks)} chunks")
    return db


def load_flat_index():
    embeddings = OpenAIEmbeddings()
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)


def query_flat_rag(question: str, db: Chroma) -> str:
    docs    = db.similarity_search(question, k=4)
    context = "\n\n".join(d.page_content for d in docs)
    resp = _oai.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user",
                   "content": ANSWER_PROMPT.format(context=context, question=question)}],
        temperature=0,
    )
    return resp.choices[0].message.content.strip()
