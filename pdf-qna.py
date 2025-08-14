# pdf_qa_cli.py
import os
import json
import shutil
from dotenv import load_dotenv
load_dotenv(override=True)

from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# ------------- CONFIG -------------
PDF_PATH = "NIPS-2017-attention-is-all-you-need-Paper.pdf"
CHROMA_PERSIST = "chroma_db"
METADATA_FILE = os.path.join(CHROMA_PERSIST, "metadata.json")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "google").lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
# ----------------------------------

if EMBEDDING_BACKEND == "google" and not os.getenv("GOOGLE_API_KEY"):
    raise SystemExit("Set GOOGLE_API_KEY before running.")

print(f"Embedding backend: {EMBEDDING_BACKEND}")
if EMBEDDING_BACKEND == "google":
    print(f"Using Google embedding model: {EMBEDDING_MODEL}")

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
elif EMBEDDING_BACKEND == "default":
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
else:
    raise ValueError(f"Unsupported EMBEDDING_BACKEND: {EMBEDDING_BACKEND}")

# 4) Check metadata to see if DB matches current settings
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

# 5) Create or load Chroma DB
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

# 7) LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

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
