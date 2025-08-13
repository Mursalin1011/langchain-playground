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
documents = loader.load() 
print(documents)