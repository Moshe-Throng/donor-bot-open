# build_index.py

from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os

load_dotenv()  # Load .env with your OpenAI key

# Set your PDF filename
pdf_path = "EU_MIP_Ethiopia_2021_2027.pdf"

# Load and split PDF into chunks
loader = PyPDFLoader(pdf_path)
pages = loader.load_and_split()

# Generate embeddings
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(pages, embeddings)

# Save index to folder
db.save_local("eu_vector_index")

print("✅ Index built and saved to 'eu_vector_index'")
