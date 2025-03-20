from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import google.generativeai as genai
from langchain_core.embeddings import Embeddings

load_dotenv()

llm = ChatGroq(
    model = "mixtral-8x7b-32768",
    temperature = 0.2,
    max_retries = 2
)

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_dir = os.path.join(current_dir,"db","chroma_db")

class GeminiEmbeddings(Embeddings):
    def __init__(self, model="models/embedding-001"):
        self.model = model

    def embed_documents(self, texts):
        """Embed a list of documents (texts)."""
        embeddings = []
        for text in texts:
            result = genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_document"  # Adjust based on your use case
            )
            embeddings.append(result["embedding"])
        return embeddings

    def embed_query(self, text):
        """Embed a single query text."""
        result = genai.embed_content(
            model=self.model,
            content=text,
            task_type="retrieval_query"  # For queries
        )
        return result["embedding"]


embeddings = GeminiEmbeddings(model="models/embedding-001")



print(f"Attempting to connect to the Chroma db at {persistent_dir}")
try:
    db = Chroma(persist_directory = persistent_dir,
            embedding_function = embeddings)
    print("Successfully Connected to Chroma DB")
except Exception as e:
    print(f"Failed to initialize Chroma DB, Error: {str(e)}")


# Define the User's Query
query = """Profits"""

retriever = db.as_retriever(
    search_type = "similarity_score_threshold",
    search_kwargs = {"k":3, "score_threshold":0.4}
)
print("Retriever initialized Successfully")

relevant_docs = retriever.invoke(query)

# Display the relevant results
print("\n---- Relevant Documents ----")
for i,doc in enumerate(relevant_docs, 1):
    print(f"Document {i}: \n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source','unknown')}\n")