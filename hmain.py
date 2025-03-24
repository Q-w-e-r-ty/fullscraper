import os
import re
import pandas as pd
import torch
import pickle
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
import google.generativeai as genai
from langchain_core.embeddings import Embeddings
from pypdf.errors import DependencyError
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Predefined metric synonyms
metric_synonyms = {
    "Sales Growth": ["revenue growth", "sales increase", "top-line growth"],
    "Capex Performance": ["capital expenditure", "investment spending", "capex trends"],
    "Order Volume and Returns": ["order bookings", "sales volume", "return rates"],
    "Client Acquisition and Retention": ["customer acquisition", "client retention", "churn rate"],
    "Debt and Loss Management": ["debt levels", "loss mitigation", "financial stability"]
}

# Custom Gemini Embeddings class
class GeminiEmbeddings(Embeddings):
    def __init__(self, model="models/embedding-001"):
        self.model = model

    def embed_documents(self, texts):
        return [genai.embed_content(model=self.model, content=text, 
                                   task_type="retrieval_document")["embedding"] for text in texts]

    def embed_query(self, text):
        return genai.embed_content(model=self.model, content=text,
                                 task_type="retrieval_query")["embedding"]

# Configure paths
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(current_dir, "Calls and Sheets")
persistent_dir = os.path.join(current_dir, "db", "finance")
os.makedirs(persistent_dir, exist_ok=True)

def standardize_quarter(filename):
    """Standardize quarter naming from filename."""
    quarter_map = {
        "q1": "Q1", "quarter 1": "Q1", "first": "Q1",
        "q2": "Q2", "quarter 2": "Q2", "second": "Q2",
        "q3": "Q3", "quarter 3": "Q3", "third": "Q3",
        "q4": "Q4", "quarter 4": "Q4", "fourth": "Q4"
    }
    filename_lower = filename.lower()
    for key, value in quarter_map.items():
        if key in filename_lower:
            return value
    return filename  # Return original if no match

def process_pdf(file_path, company, doc_type, quarter):
    """Process a single PDF file with error handling."""
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        for doc in documents:
            doc.metadata.update({
                "company": company,
                "file_type": doc_type,
                "quarter": quarter,
                "source": file_path
            })
        return documents
    except DependencyError:
        logger.warning(f"Skipping encrypted PDF: {file_path}")
        return []
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return []

# Embed documents if vector store is empty
if not os.listdir(persistent_dir):
    logger.info("Initializing vector store...")
    all_docs = []
    
    for doc_type in ["Calls", "Sheets"]:
        type_dir = os.path.join(base_dir, doc_type)
        if not os.path.exists(type_dir):
            continue

        for company in os.listdir(type_dir):
            company_dir = os.path.join(type_dir, company)
            if not os.path.isdir(company_dir):
                continue

            for pdf_file in os.listdir(company_dir):
                if not pdf_file.endswith(".pdf"):
                    continue
                
                quarter = standardize_quarter(os.path.splitext(pdf_file)[0])
                file_path = os.path.join(company_dir, pdf_file)
                
                documents = process_pdf(file_path, company,
                                      "concall" if doc_type == "Calls" else "sheet",
                                      quarter)
                if documents:
                    text_splitter = CharacterTextSplitter(
                        chunk_size=1500,  # Increased for more context
                        chunk_overlap=150,
                        add_start_index=True
                    )
                    split_docs = text_splitter.split_documents(documents)
                    all_docs.extend(split_docs)

    if not all_docs:
        raise FileNotFoundError("No processable PDF files found in directory structure")

    logger.info("\n---- Document Processing Summary ----")
    logger.info(f"Total chunks created: {len(all_docs)}")
    logger.info(f"Companies processed: {sorted(set(d.metadata['company'] for d in all_docs))}")
    logger.info(f"Quarters processed: {sorted(set(d.metadata['quarter'] for d in all_docs))}")
    logger.info(f"Sample chunk metadata: {all_docs[0].metadata}\n")

    embeddings = GeminiEmbeddings()
    logger.info("\n---- Creating Vector Store ----")
    db = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=persistent_dir
    )
    logger.info("Vector store created and persisted successfully!")
else:
    logger.info("Vector store already exists.")

# Financial Query Engine
class FinancialQueryEngine:
    def __init__(self):
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.persistent_dir = os.path.join(self.current_dir, "db", "finance")
        self.embeddings = GeminiEmbeddings()
        self.db = Chroma(persist_directory=self.persistent_dir, embedding_function=self.embeddings)
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        logger.info("FinBERT initialized successfully.")

    def load_company_keywords(self, company):
        """Load company-specific keywords from a text file."""
        keyword_file = os.path.join(self.current_dir, f"{company.lower()}.txt")
        if os.path.exists(keyword_file):
            with open(keyword_file, 'r') as f:
                return [line.strip() for line in f.readlines() if line.strip()]
        logger.warning(f"Keyword file for {company} not found.")
        return []

    def extract_relevant_text(self, content, metric):
        """Extract text relevant to the metric."""
        keywords = metric.lower().split(" and ") + metric.lower().split()
        relevant_sentences = [s for s in content.split('.') if any(k in s.lower() for k in keywords)]
        return '. '.join(relevant_sentences) if relevant_sentences else content

    def get_sentiment_scores(self, text):
        """Get sentiment probabilities from FinBERT."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]  # [positive, neutral, negative]
        return probs

    def get_quarterly_scores(self, company, metric, year=None):
        """Retrieve quarterly scores and sentiment tensors."""
        company_keywords = self.load_company_keywords(company)
        synonyms = metric_synonyms.get(metric, [])
        specific_query = f"{company} {metric} {' '.join(synonyms)} {' '.join(company_keywords[:5])}"
        general_query = f"{company} {' '.join(company_keywords[:10])} financial performance"
        filter_dict = {"company": company}
        k = 15  # Increased retrieval count for robustness

        docs = self.db.similarity_search(specific_query, k=k, filter=filter_dict)
        if not docs:
            docs = self.db.similarity_search(general_query, k=k, filter=filter_dict)
            if not docs:
                logger.warning(f"No documents for {company} - {metric}")
                return {"Q1": 0.5, "Q2": 0.5, "Q3": 0.5, "Q4": 0.5}, [], False  # Neutral default

        quarterly_scores = {"Q1": [], "Q2": [], "Q3": [], "Q4": []}
        all_tensors = []
        for doc in docs:
            quarter = doc.metadata.get("quarter", "Q1")  # Default to Q1 if missing
            relevant_text = self.extract_relevant_text(doc.page_content, metric)
            probs = self.get_sentiment_scores(relevant_text)
            score = (probs[0] * 1.0 + probs[1] * 0.5 + probs[2] * 0.0).item() or 0.5  # Minimum 0.5
            all_tensors.append((quarter, probs))
            if quarter in quarterly_scores:
                quarterly_scores[quarter].append(score)

        result = {q: sum(scores) / len(scores) if scores else 0.5 for q, scores in quarterly_scores.items()}
        return result, all_tensors, True

def generate_score_report():
    """Generate performance report and save tensors."""
    engine = FinancialQueryEngine()
    companies = ["HCL", "Wipro", "TCS", "Infosys"]
    metrics = [
        "Capex Performance", "Order Volume and Returns", "Sales Growth",
        "Client Acquisition and Retention", "Debt and Loss Management"
    ]
    year = "2023"
    
    data = []
    all_sentiment_data = []

    for company in companies:
        logger.info(f"Analyzing {company}")
        for metric in metrics:
            quarterly_scores, tensors, doc_found = engine.get_quarterly_scores(company, metric, year)
            print(f"\nCompany: {company}\nMetric: {metric}\n- Documents Found: {'Yes' if doc_found else 'No'}")
            print("- Quarterly Scores:")
            for q, score in quarterly_scores.items():
                print(f"  {q}: {score:.3f}")
            valid_scores = [s for s in quarterly_scores.values() if s > 0.5]
            overall_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.5
            data.append({
                "Company": company, "Metric": metric, "Documents Found": doc_found,
                **quarterly_scores, "Overall Score": overall_score
            })
            all_sentiment_data.extend([(company, metric, q, t) for q, t in tensors])

    df = pd.DataFrame(data)
    df.to_csv("quarterly_performance_report.csv", index=False)
    with open("sentiment_tensors.pkl", "wb") as f:
        pickle.dump(all_sentiment_data, f)
    logger.info("Report and tensors saved.")

if __name__ == "__main__":
    generate_score_report()