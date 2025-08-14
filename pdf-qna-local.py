# pdf_qa_cli.py

import os
import json
import shutil
from dotenv import load_dotenv
load_dotenv(override=True)

from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama

# Enable LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
# Must be set in your .env file or environment
# os.environ["LANGCHAIN_API_KEY"] = "YOUR_LANGSMITH_API_KEY"
os.environ.setdefault("LANGCHAIN_PROJECT", "Local PDF RAG with Ollama")

# ------------- CONFIG -------------
PDF_PATH = "NIPS-2017-attention-is-all-you-need-Paper.pdf"
CHROMA_PERSIST = "chroma_db"
METADATA_FILE = os.path.join(CHROMA_PERSIST, "metadata.json")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "huggingface").lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")  # HF default
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2:7b")  # Local LLM model name
# ----------------------------------

print(f"Embedding backend: {EMBEDDING_BACKEND}")
if EMBEDDING_BACKEND == "google":
    if not os.getenv("GOOGLE_API_KEY"):
        raise SystemExit("Set GOOGLE_API_KEY before running.")
    print(f"Using Google embedding model: {EMBEDDING_MODEL}")
else:
    print(f"Using HuggingFace embedding model: {EMBEDDING_MODEL}")

# 1) Load PDF
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

# 2) Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
chunks = splitter.split_documents(documents)
print(f"Loaded {len(documents)} pages → {len(chunks)} chunks")

# 3) Select embeddings
if EMBEDDING_BACKEND == "google":
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
elif EMBEDDING_BACKEND == "huggingface":
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
else:
    raise ValueError(f"Unsupported EMBEDDING_BACKEND: {EMBEDDING_BACKEND}")

# 4) Metadata check
def needs_rebuild():
    if not os.path.exists(METADATA_FILE):
        return True
    try:
        with open(METADATA_FILE, "r") as f:
            meta = json.load(f)
        return meta.get("backend") != EMBEDDING_BACKEND or meta.get("model") != EMBEDDING_MODEL
    except Exception:
        return True

def save_metadata():
    os.makedirs(CHROMA_PERSIST, exist_ok=True)
    with open(METADATA_FILE, "w") as f:
        json.dump({"backend": EMBEDDING_BACKEND, "model": EMBEDDING_MODEL}, f)

# 5) Create/load Chroma DB
if needs_rebuild():
    if os.path.exists(CHROMA_PERSIST):
        shutil.rmtree(CHROMA_PERSIST)
    db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PERSIST)
    save_metadata()
    print("Built and persisted Chroma DB.")
else:
    db = Chroma(persist_directory=CHROMA_PERSIST, embedding_function=embeddings)
    print("Loaded Chroma DB from disk.")

# 6) Retriever
retriever = db.as_retriever(search_kwargs={"k": TOP_K})

# 7) LLM - Ollama local model
llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0,
    num_ctx=4096,  # Context window
)

# 8) RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# 9) CLI loop
print("PDF Q&A CLI ready. Type a question (or 'exit').")
while True:
    q = input("> ").strip()
    if q.lower() in ("exit", "quit", "q"):
        break
    res = qa.invoke({"query": q})
    answer = res["result"] if isinstance(res, dict) else res
    print("\n== Answer ==\n")
    print(answer)
    if isinstance(res, dict) and res.get("source_documents"):
        print("\n-- Source documents (top chunks) --")
        for d in res["source_documents"]:
            src = d.metadata.get("source", "<no-source>")
            page = d.metadata.get("page", d.metadata.get("page_number", "-"))
            print(f"• {src} (page: {page})")
    print("\n-----------------\n")
