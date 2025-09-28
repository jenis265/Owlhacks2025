import numpy as np
import pandas as pd
import requests
import re 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import sys
from deep_translator import GoogleTranslator 
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils.class_weight import compute_class_weight
import joblib
import pymongo as MongoClient
import ast
from sklearn.metrics import f1_score



# -----------------------
# 1. Load external FoodData (lookup table)
# -----------------------
df2 = pd.read_csv(r"C:\Users\tuo60501\Downloads\FoodData.csv")
df2['Food'] = df2['Food'].str.lower().str.strip()
df2['Allergy'] = df2['Allergy'].str.strip()

food_to_allergy = dict(zip(df2['Food'], df2['Allergy']))  # mapping

# -----------------------
# 2. Function to fetch OpenFoodFacts API Data with pagination
# -----------------------
def fetch_openfoodfacts(max_pages=20, page_size=100):
    """Fetch multiple pages from OpenFoodFacts API."""
    url = "https://world.openfoodfacts.org/cgi/search.pl"
    all_products = []
    
    for page in range(1, max_pages + 1):
        params = {
            "action": "process",
            "tagtype_0": "countries",
            "tag_contains_0": "contains",
            "tag_0": "united-states",
            "fields": "code,product_name,brands,ingredients_text_en,allergens_tags,categories_tags,image_url,nutriments", 
            "page_size": page_size,
            "page": page,
            "json": 1
        }
        data = requests.get(url, params=params).json()
        products = data.get("products", [])
        if not products:  # stop if empty page
            break
        all_products.extend(products)
        print(f"Page {page} → pulled {len(products)} products")
    
    return all_products

# Fetch (adjust max_pages as needed)
products = fetch_openfoodfacts(max_pages=20, page_size=100)  # up to ~10k rows
print("Total products collected:", len(products))

# -----------------------
# 3. Helper Cleaning Functions
# -----------------------
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
        return GoogleTranslator(source='auto', target='en').translate(text).lower()
    except Exception:
        return text

# -----------------------
# 4. Build Clean DataFrame from API
# -----------------------
cleaned = []
for p in products:
    raw_ing = extract_ingredients(p.get("ingredients_text_en"))
    clean_ing = clean_ingredients_text(raw_ing)
    if clean_ing:
        cleaned.append({
            "code": p.get("code"),
            "product_name": p.get("product_name"),
            "brands": p.get("brands"),
            "ingredients_clean": clean_ing,
            "allergens": p.get("allergens_tags"),
            "categories": p.get("categories_tags"),
            "image_url": p.get("image_url"),
            "nutriments": p.get("nutriments"),
        })

df = pd.DataFrame(cleaned)
print("After filtering:", len(df), "clean EN products")

# Normalise allergen tags
def clean_allergen_tag(tag):
    if not isinstance(tag, str):
        return tag
    return re.sub(r'^[a-z]{2,3}:', '', tag)

df['allergens_clean'] = df['allergens'].apply(
    lambda tags: [clean_allergen_tag(tag) for tag in tags] if isinstance(tags, list) else []
)

# -----------------------
# 5. Translate + Allergy Mapping from external FoodData
# -----------------------
df["ingredients_translated"] = df["ingredients_clean"].apply(translate_to_en)

def match_allergies_extended(ingredients_str, mapping):
    if not isinstance(ingredients_str, str):
        return []
    allergies = []
    for food, allergy in mapping.items():
        if food in ingredients_str:  # substring check
            allergies.append(allergy)
    return list(set(allergies))

df['allergy'] = df['ingredients_translated'].apply(lambda x: match_allergies_extended(x, food_to_allergy))

# -----------------------
# 6. Inspect & Counts
# -----------------------
'''
empty_count = df['allergy'].apply(lambda x: len(x) == 0).sum()
non_empty_count = len(df) - empty_count
print("Empty allergy rows:", empty_count)
print("Non-empty allergy rows:", non_empty_count)
'''


#print(df[['product_name','ingredients_translated','allergy']].head(10))



def clean_allergy_entry(x):
    # Case 1: missing value
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []

    # Case 2: numpy array
    if isinstance(x, np.ndarray):
        return [str(i) for i in x.tolist()]

    # Case 3: string that looks like a list
    if isinstance(x, str):
        try:
            parsed = ast.literal_eval(x)   # safely parse "['milk','nuts']"
            if isinstance(parsed, (list, set, tuple, np.ndarray)):
                return [str(i) for i in parsed]
            else:
                return [parsed]
        except Exception:
            return [x]   # fallback: wrap raw string

    # Case 4: already a list/set/tuple
    if isinstance(x, (list, set, tuple)):
        return [str(i) for i in x]

    # Fallback: cast to string inside a list
    return [str(x)]

# Apply cleaning
df['allergy'] = df['allergy'].apply(clean_allergy_entry)



# -----------------------
# 1. Only use labeled rows
# -----------------------
train_df = df[df['allergy'].map(len) > 0]   # labeled rows

# -----------------------
# 2. Train/validation split
# -----------------------
X_text = train_df["ingredients_translated"]
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(train_df["allergy"])

X_train_text, X_val_text, Y_train, Y_val = train_test_split(
    X_text, Y, test_size=0.2, random_state=42
)

# -----------------------
# 3. Vectorize text
# -----------------------
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_train = vectorizer.fit_transform(X_train_text)
X_val   = vectorizer.transform(X_val_text)

# -----------------------
# 4. Define multi-output XGBoost
# -----------------------
xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss"
)
multi_xgb = MultiOutputClassifier(xgb, n_jobs=-1)

# -----------------------
# 5. Optional: GridSearch
# -----------------------
param_grid = {
    "estimator__n_estimators": [200],
    "estimator__max_depth": [4],
    "estimator__learning_rate": [0.1],
    "estimator__subsample": [0.8],
    "estimator__colsample_bytree": [0.8]
}

grid = GridSearchCV(
    multi_xgb,
    param_grid,
    cv=3,
    scoring="f1_macro",
    verbose=2,
    n_jobs=-1
)

grid.fit(X_train, Y_train)

print("\nBest Params =", grid.best_params_)
print("Best CV Score =", grid.best_score_)

Y_val_pred = grid.best_estimator_.predict(X_val)

val_f1_macro = f1_score(Y_val, Y_val_pred, average="macro", zero_division=0)
val_f1_micro = f1_score(Y_val, Y_val_pred, average="micro", zero_division=0)

print("\nValidation F1 Scores:")
print("Macro F1:", val_f1_macro)
print("Micro F1:", val_f1_micro)

# Optional: inspect some predictions
val_labels_pred = mlb.inverse_transform(Y_val_pred)
for true, pred in zip(train_df["allergy"].iloc[:5], val_labels_pred[:5]):
    print("True:", true, "Pred:", pred)

# -----------------------
# 7. Save vectorizer, model, mlb
# -----------------------
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(grid.best_estimator_, "best_xgb_model.pkl")
joblib.dump(mlb, "mlb.pkl")





