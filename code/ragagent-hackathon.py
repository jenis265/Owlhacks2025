import os
import re
import joblib
import requests
import pandas as pd
from typing import List

from google import genai 
from pymongo import MongoClient

# LangChain & Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker

# API Framework
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# -----------------------
# 1. Configure Gemini
# -----------------------

def gemini_llm(prompt: str, context: str) -> str:
    """Send prompt + context to Gemini."""
    model = genai.GenerativeModel("gemini-pro", api_key="AIzaSyD3eSzBViF98HFwvrHSbRGUFWU-sG0NFg8")
    full_prompt = (
        "You are a helpful food allergy assistant.\n"
        "Use ONLY the context to answer.\n"
        "If the answer is not in the context, say 'I don't know.'\n\n"
        f"Context:\n{context}\n\nQuestion: {prompt}"
    )
    response = model.generate_content(full_prompt)
    return response.text

# -----------------------
# 2. MongoDB for product metadata
# -----------------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
client = MongoClient(MONGO_URI)
db = client["allergyDB"]
collection = db["products"]

# -----------------------
# 3. Load ML model artifacts
# -----------------------
vectorizer = joblib.load("vectorizer.pkl")   # TF-IDF
mlb = joblib.load("mlb.pkl")                 # MultiLabelBinarizer
multi_clf = joblib.load("best_xgb_model.pkl")     # trained XGB model

def enrich_allergens_with_ml(code: str, ingredients: str, allergens: list):
    """
    If allergens are missing/empty, predict using trained ML model.
    Returns a dict with 'allergens_real' and 'allergens_predicted'.
    """
    if allergens and len(allergens) > 0:
        return {"allergens_real": allergens, "allergens_predicted": []}
    
    if not ingredients or ingredients.strip() == "":
        return {"allergens_real": [], "allergens_predicted": []}
    
    # Transform with TF-IDF
    X_new = vectorizer.transform([ingredients])
    
    # Predict with XGB multi-output model
    Y_pred = multi_clf.predict(X_new)
    
    predicted_labels = [mlb.classes_[i] for i, v in enumerate(Y_pred[0]) if v == 1]
    return {"allergens_real": [], "allergens_predicted": predicted_labels}

# -----------------------
# 4. Chroma for embeddings
# -----------------------
EMBED_MODEL = HuggingFaceBgeEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True}
)

persist_dir = "chroma_store"
vectorstore = None

def build_chroma_from_mongo():
    """Build Chroma vector DB from MongoDB ingredients field."""
    docs = []
    metadatas = []
    for record in collection.find({"ingredients_translated": {"$exists": True}}):
        docs.append(record["ingredients_translated"])
        metadatas.append({
            "product_name": record.get("product_name", ""),
            "code": record.get("code", ""),
            "allergens": record.get("allergens", []),
            "brands": record.get("brands", ""),
        })
    vs = Chroma.from_texts(
        docs,
        embedding=EMBED_MODEL,
        persist_directory=persist_dir,
        metadatas=metadatas
    )
    vs.persist()
    return vs

# initialize vectorstore if not already persisted
if os.path.isdir(persist_dir) and os.listdir(persist_dir):
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=EMBED_MODEL)
else:
    vectorstore = build_chroma_from_mongo()

# -----------------------
# 5. Retriever + reranker
# -----------------------
cross = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

def build_retriever(vectordb, k_candidates=10, k_final=5):
    base = vectordb.as_retriever(search_kwargs={"k": k_candidates})
    reranker = CrossEncoderReranker(model=cross, top_n=k_final)
    return ContextualCompressionRetriever(base_retriever=base, base_compressor=reranker)

retriever = build_retriever(vectorstore)

# -----------------------
# 6. RAG QA Function (with ML enrichment)
# -----------------------
def rag_answer(question: str) -> str:
    docs = retriever.get_relevant_documents(question)
    
    context_parts = []
    
    for d in docs:
        product_name = d.metadata.get("product_name", "")
        code = d.metadata.get("code", "")
        
        # Pull full record from Mongo
        record = collection.find_one({"code": code}) or {}
        ingredients = record.get("ingredients_translated", "")
        allergens = record.get("allergens", [])
        
        # 🔄 Merge predictions if needed
        enriched = enrich_allergens_with_ml(code, ingredients, allergens)
        
        allergens_final = []
        if enriched["allergens_real"]:
            allergens_final.extend(enriched["allergens_real"])
        if enriched["allergens_predicted"]:
            allergens_final.extend(enriched["allergens_predicted"])
        
        # Build context snippet
        context_snippet = (
            f"Product: {product_name}\n"
            f"Ingredients: {ingredients}\n"
            f"Allergens: {', '.join(allergens_final) if allergens_final else 'None found'}"
        )
        context_parts.append(context_snippet)
    
    # Combine docs into context
    context = "\n\n".join(context_parts)
    
    # Call Gemini
    answer = gemini_llm(question, context)
    
    return answer

# -----------------------
# 7. FastAPI backend
# -----------------------
app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask(query: Query):
    answer = rag_answer(query.question)
    return {"answer": answer}

# -----------------------
# 8. Run server
# -----------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
