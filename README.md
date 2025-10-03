# Owlhacks2025
Welcome to Action-Reaction! This is our OwlHacks 2025 Hackathon project. Below you will find a description of the project and how to run it on your machine!

# Action-Reaction 🥜🚫

Your AI-powered allergy-safe product scanner. Scan barcodes, get instant allergen information, and shop with confidence.

## Features
- 📷 Barcode scanning (camera, image upload, or manual entry)
- 🤖 AI chatbot for allergy questions
- 🛒 Smart shopping lists
- 👨‍🍳 Recipe generator (allergen-safe)
- 🧠 ML-powered allergen prediction using XGBoost

## Quick Start

### Prerequisites
- Python 3.8 or higher
- A modern web browser (Chrome, Firefox, Safari)

### Setup & Run

1. **Download the files**
   - `ragagent-hackathon.py` (backend server) (in the "code" file)
   - `withapi.html` (frontend)
   - `vectorizer.pkl`, `best_xgb_model.pkl`, `mlb.pkl` (ML models) (also in the "code" file)

2. **Install Python packages**
   ```bash
   pip install fastapi uvicorn python-multipart pandas numpy scikit-learn xgboost joblib deep-translator google-generativeai langchain langchain-community chromadb sentence-transformers pillow pyzbar requests

3. **Start the backend server**
  python ragagent-hackathon.py
  Keep this terminal window open. You should see:

  INFO: Uvicorn running on http://0.0.0.0:7860

5. **Start the frontend server**
  Open a new terminal window and run:
  python -m http.server 8000

6. **Open in browser**
  Go to: http://localhost:8000/withapi.html

### Usage 
Login:
Username: testuser
Password: password123
Or create your own account! You can use any username or password that you like.

**Try These Features**
- Scan a product: Try barcode 3017620422003 (Nutella) or scan products from your pantry
- Add allergens: Go to "Allergens" tab and add your allergies
- Ask the chatbot: "What products are safe for someone with a peanut allergy?"
- Generate recipes: Add food items to your shopping list and click "Generate Recipes"
**Test Barcodes**
3017620422003 - Nutella
5000159484695 - Heinz Ketchup
Or use any barcode from products you have at home!

**Troubleshooting**
"Could not connect to chatbot server"
- Make sure ragagent-hackathon.py is running in a separate terminal
- Check that it shows port 7860

Camera not working?
- Allow camera permissions when prompted
- Try uploading an image instead

Barcode not found?
- Not all products are in the OpenFoodFacts database
- Try products you have at home for best results

### Tech Stack
Frontend: HTML, JavaScript, Tailwind CSS
Backend: FastAPI, Python
ML: XGBoost, scikit-learn
AI: Google Gemini API, RAG with ChromaDB
Data: OpenFoodFacts API

### Team
Built during OwlHacks2025 by Jeni Sorathiya, Evania Bhattarai, Maahin Mirza, Massimo Camuso, Tyler Baughman, and Avinsh Saini
   
