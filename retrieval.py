from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

llm = ChatGroq(
    model = "mixtral-8x7b-32768",
    temperature = 0.2,
    max_retries = 2
)

current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_dir = os.path.join(current_dir,"db","chroma_db")

embeddings = HuggingFaceEmbeddings(model = "")
