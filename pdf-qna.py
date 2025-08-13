# pdf_qa_cli.py
import os
from dotenv import load_dotenv
load_dotenv(override=True)  # load .env file if exists

# 1) LangChain + Google GenAI imports
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
# LangChain<->Google wrapper
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# chain helper (simple QA)
from langchain.chains import RetrievalQA

# ------------- CONFIG -------------
PDF_PATH = "NIPS-2017-attention-is-all-you-need-Paper.pdf"            # <-- change to your file
CHROMA_PERSIST = "chroma_db"      # local persistence dir
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4                         # number of chunks to retrieve
# ----------------------------------

# 0) sanity check env
if not os.getenv("GOOGLE_API_KEY"):
    raise SystemExit("Set GOOGLE_API_KEY (see Google AI Studio) before running.")

# 1) load PDF
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()   # list of Document objects (each page usually)

# 2) split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
chunks = splitter.split_documents(documents)   # list[Document]

print(f"Loaded {len(documents)} pages → {len(chunks)} chunks")

# 3) embeddings using Gemini embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")  # or omit model to use default

# 4) create or load Chroma vectorstore
if not os.path.exists(CHROMA_PERSIST):
    db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PERSIST)
    print("Built and persisted Chroma DB.")
else:
    db = Chroma(persist_directory=CHROMA_PERSIST, embedding_function=embeddings)
    print("Loaded Chroma DB from disk.")

# 5) retriever
retriever = db.as_retriever(search_kwargs={"k": TOP_K})

# 6) LLM (Gemini chat)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")  # pick the model you want

# 7) build a simple RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# 8) CLI loop
print("PDF Q&A CLI ready. Type a question (or 'exit').")
while True:
    q = input("> ").strip()
    if q.lower() in ("exit", "quit", "q"):
        break
    res = qa.invoke({"query": q})
    answer = res["result"] if isinstance(res, dict) else res
    print("\n== Answer ==\n")
    print(answer)
    # show sources if available
    if isinstance(res, dict) and res.get("source_documents"):
        print("\n-- Source documents (top chunks) --")
        for d in res["source_documents"]:
            src = d.metadata.get("source", "<no-source>")
            page = d.metadata.get("page", d.metadata.get("page_number", "-"))
            print(f"• {src} (page: {page})")
    print("\n-----------------\n")
