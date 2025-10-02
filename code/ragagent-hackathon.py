import os
import requests
from typing import List

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import re
from deep_translator import GoogleTranslator
import joblib
import xgboost

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker

from PIL import Image
from pyzbar.pyzbar import decode
import io
import numpy as np
from typing import Optional
# -----------------------
# 1. Gemini API
# -----------------------
import google.generativeai as genai
import os
from pprint import pprint



def extract_ingredients(ingredients_field):
    if not ingredients_field:
        return None
    if isinstance(ingredients_field, list):
        en = next((i.get("text") for i in ingredients_field if i.get("lang") == "en"), None)
        if en:
            return en
        main = next((i.get("text") for i in ingredients_field if i.get("lang") == "main"), None)
        return main
    if isinstance(ingredients_field, str):
        return ingredients_field
    return None

def clean_ingredients_text(text):
    if not text:
        return None
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def translate_to_en(text):
    if not isinstance(text, str) or text.strip() == "":
        return text
    try:
        # Note: Requires `deep-translator` library
        return GoogleTranslator(source='auto', target='en').translate(text).lower()
    except Exception:
        return text


# 1. Gemini API

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyD3eSzBViF98HFwvrHSbRGUFWU-sG0NFg8")

# Configure the API key once at the start of your script
genai.configure(api_key=GEMINI_API_KEY)

def gemini_llm(prompt: str, context: str) -> str:
    # Do not pass the api_key here; it's already configured
    model = genai.GenerativeModel("models/gemini-2.5-flash")

    full_prompt = (
        "You are a helpful food allergy assistant.\n"
        "Use ONLY the context to answer.\n"
        "If the answer is not in the context, say 'I don't know.'\n\n"
        f"Context:\n{context}\n\nQuestion: {prompt}"
    )

    response = model.generate_content(full_prompt)
    return response.text

# -----------------------
# 2. Load OpenFoodFacts via API
# -----------------------
def fetch_off_data(page_limit=5):
    all_docs = []
    url = "https://world.openfoodfacts.org/cgi/search.pl"
    for page in range(1, page_limit + 1):
        params = {
            "action": "process",
            "tagtype_0": "countries",
            "tag_contains_0": "contains",
            "tag_0": "united-states",
            "fields": "code,product_name,brands,ingredients_text_en,allergens_tags",
            "page_size": 100,
            "page": page,
            "json": 1
        }
        r = requests.get(url, params=params)
        try:
            data = r.json()
            products = data.get("products", [])
        except requests.exceptions.JSONDecodeError:
            print(f"Error decoding JSON for page {page}. Skipping.")
            continue
            
        print(f"Fetched {len(products)} products from page {page}.")

        for p in products:
            # Use the extract function for robustness
            ingredients = extract_ingredients(p.get("ingredients_text_en", ""))
            
            if not ingredients:
                continue

            cleaned_ingredients = clean_ingredients_text(ingredients)
            translated_ingredients = translate_to_en(cleaned_ingredients)
            
            if translated_ingredients:
                all_docs.append({
                    "text": translated_ingredients,
                    "product_name": p.get("product_name", ""),
                    "code": p.get("code", ""),
                    "allergens": ", ".join(p.get("allergens_tags", [])) or None,
                    "brands": p.get("brands", "")
                })
    return all_docs

# -----------------------
# 3. Chroma vectorstore
# -----------------------
EMBED_MODEL = HuggingFaceBgeEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True}
)

persist_dir = "chroma_store"

def build_chroma_from_api(page_limit=5):
    docs = fetch_off_data(page_limit)
    texts = [d["text"] for d in docs]
    metadatas = []
    for d in docs:
        # Ensure metadata values are str, int, float, bool, or None
        metadatas.append({
            "product_name": d["product_name"] or None,
            "code": d["code"] or None,
            "allergens": d["allergens"] or None,
            "brands": d["brands"] or None
        })
    vs = Chroma.from_texts(
        texts=texts,
        embedding=EMBED_MODEL,
        persist_directory=persist_dir,
        metadatas=metadatas
    )
    vs.persist()
    return vs

# initialize vectorstore
if os.path.isdir(persist_dir) and os.listdir(persist_dir):
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=EMBED_MODEL)
else:
    vectorstore = build_chroma_from_api(page_limit=5)

# -----------------------
# 4. Retriever + reranker
# -----------------------
cross = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

def build_retriever(vectordb, k_candidates=10, k_final=10):
    base = vectordb.as_retriever(search_kwargs={"k": k_candidates})
    reranker = CrossEncoderReranker(model=cross, top_n=k_final)
    return ContextualCompressionRetriever(base_retriever=base, base_compressor=reranker)

retriever = build_retriever(vectorstore)


try:
    vectorizer = joblib.load("vectorizer.pkl")
    best_xgb_model = joblib.load("best_xgb_model.pkl")
    mlb = joblib.load("mlb.pkl")
    print("Successfully loaded XGBoost model, vectorizer, and MultiLabelBinarizer.")
except FileNotFoundError:
    print("Warning: Could not find one or more model files. Prediction functionality will be unavailable.")
    vectorizer, best_xgb_model, mlb = None, None, None


def predict_allergens(ingredients_text: str):
    """
    Predicts allergens using the loaded XGBoost model.
    Assumes `vectorizer`, `best_xgb_model`, and `mlb` are loaded.
    """
    if not all([vectorizer, best_xgb_model, mlb]):
        print("Model components not loaded. Skipping prediction.")
        return []

    # Vectorize the input text
    ingredients_vectorized = vectorizer.transform([ingredients_text])

    # Make predictions
    predictions = best_xgb_model.predict(ingredients_vectorized)
    
    # Inverse transform to get human-readable labels
    predicted_allergens = mlb.inverse_transform(predictions)
    
    # The output is a list of tuples, so we flatten it
    return [allergen for sublist in predicted_allergens for allergen in sublist]

# -----------------------
# 5. RAG QA Function
# -----------------------
def rag_answer(question: str) -> str:
    docs = retriever.invoke(question)

    print("\n--- Retrieved Documents ---")
    context_parts = []
    for doc in docs:
        product_name = doc.metadata.get("product_name", "")
        ingredients = doc.page_content
        allergens = doc.metadata.get("allergens")

        # If allergens are missing, predict them
        if not allergens or allergens == "None found":
            if ingredients:
                predicted_allergens = predict_allergens(ingredients)
                if predicted_allergens:
                    allergens = ", ".join(predicted_allergens)
                    print(f"--> XGBoost prediction used for product '{product_name}'")
                else:
                    allergens = "None found (prediction failed)"

        # Prepare context snippet
        context_snippet = f"Product: {product_name}\nIngredients: {ingredients}\nAllergens: {allergens}"
        context_parts.append(context_snippet)

        print(f"Product: {product_name}")
        print(f"Ingredients: {ingredients[:150]}...")
        print(f"Allergens: {allergens}")
        print("-" * 20)

    print("--- End Retrieved Documents ---\n")
    context = "\n\n".join(context_parts)
    answer = gemini_llm(question, context)
    return answer



# -----------------------
# 6. FastAPI backend
# -----------------------
app = FastAPI()

# CORS middleware MUST be added BEFORE any routes are defined
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask(query: Query):
    answer = rag_answer(query.question)
    return {"answer": answer}

@app.post("/check_barcode")
async def check_barcode(
    barcode: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    """
    Check barcode either from:
    1. Direct barcode string (manual entry)
    2. Uploaded image (extract barcode from image)
    """
    detected_barcode = None
    
    # Option 1: Manual barcode entry
    if barcode:
        detected_barcode = barcode.strip()
    
    # Option 2: Extract barcode from uploaded image
    elif image:
        try:
            contents = await image.read()
            img = Image.open(io.BytesIO(contents)).convert("RGB")
            img_np = np.array(img)
            barcodes = decode(img_np)
            
            if barcodes:
                detected_barcode = barcodes[0].data.decode("utf-8")
        except Exception as e:
            return {
                "found": False,
                "message": f"Error processing image: {str(e)}"
            }
    
    if not detected_barcode:
        return {
            "found": False,
            "message": "No barcode provided or detected."
        }
    
    # Fetch product from OpenFoodFacts API
    try:
        url = f"https://world.openfoodfacts.org/api/v0/product/{detected_barcode}.json"
        response = requests.get(url)
        data = response.json()
        
        if data.get("status") == 1:
            product = data.get("product", {})
            
            # Extract and clean ingredients
            raw_ingredients = extract_ingredients(product.get("ingredients_text_en"))
            cleaned_ingredients = clean_ingredients_text(raw_ingredients)
            translated_ingredients = translate_to_en(cleaned_ingredients)
            
            # Get allergens
            allergens_tags = product.get("allergens_tags", [])
            allergens_list = [tag.replace("en:", "") for tag in allergens_tags]
            
            # If no allergens in the product data, predict using XGBoost
            if not allergens_list and translated_ingredients and all([vectorizer, best_xgb_model, mlb]):
                predicted_allergens = predict_allergens(translated_ingredients)
                if predicted_allergens:
                    allergens_list = predicted_allergens
                    print(f"--> XGBoost prediction used for barcode {detected_barcode}")
            
            return {
                "found": True,
                "code": detected_barcode,
                "product_name": product.get("product_name", "Unknown Product"),
                "brands": product.get("brands", ""),
                "ingredients": translated_ingredients or "No ingredients found",
                "allergens": allergens_list,
                "image_url": product.get("image_url", "")
            }
        else:
            return {
                "found": False,
                "message": f"Product with barcode {detected_barcode} not found in database."
            }
    except Exception as e:
        return {
            "found": False,
            "message": f"Error fetching product: {str(e)}"
        }

# -----------------------
# 7. Run server
# -----------------------
if __name__ == "__main__":
    # ... your test calls ...
    uvicorn.run(app, host="0.0.0.0", port=7860)

