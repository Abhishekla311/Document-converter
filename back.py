import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.sequence import pad_sequences

from exception.CustomException import CustomException
from logger.logger import get_logger
from src.train import ModelTrainerPipeline

logger = get_logger(__name__)

app = FastAPI(title="Multi-Model RNN & CNN Predictor API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# --- Templates सेटअप (HTML रेंडर करने के लिए) ---
# ध्यान दें: यह 'backend' रूट फोल्डर के अंदर से 'templates' को ढूंढेगा
templates = Jinja2Templates(directory="templates")

ARTIFACTS_DIR = os.path.join("backend", "artifacts")
word2idx = {}
idx2word = {}
vocab_size = 0
models = {}

class TextRequest(BaseModel):
    text: str

@app.on_event("startup")
def startup_event():
    global word2idx, idx2word, vocab_size, models
    try:
        logger.info("FastAPI Server Startup: Loading Metadata and Models...")
        trainer = ModelTrainerPipeline()
        word2idx, idx2word, vocab_size = trainer.build_custom_vocabulary()
        
        model_names = {
            "m11": "m11.h5",
            "m1M": "m1M.h5",
            "mM1": "mM1.h5",
            "mMM": "mMM.h5",
            "mCombine": "mCombine.h5"
        }
        
        for key, filename in model_names.items():
            model_path = os.path.join(ARTIFACTS_DIR, filename)
            if os.path.exists(model_path):
                models[key] = load_model(model_path)
                logger.info(f"Successfully loaded model: {filename}")
            else:
                logger.warning(f"⚠️ Model file missing: {filename}")
                
        logger.info("🎉 FastAPI Server ready for predictions!")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise CustomException(e, sys)

def text_to_sequence(text: str, max_len: int):
    tokens = text.replace(',', ' ').split()
    seq = [word2idx.get(w, word2idx.get("<OOV>", 1)) for w in tokens]
    padded = pad_sequences([seq], maxlen=max_len, padding='post', value=word2idx.get("<PAD>", 0))
    return padded

def preprocess_uploaded_image(img_path: str):
    # target_size को tuple के रूप में (64, 64) दें, (64, 64, 3) नहीं
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    
    # स्केलिंग सुनिश्चित करें (0 से 1 के बीच)
    img_array = img_array / 255.0
    
    # बैच डाइमेंशन जोड़ें ताकि शेप (1, 64, 64, 3) हो जाए
    img_array = np.expand_dims(img_array, axis=0)
    return img_array



# =====================================================================
#  UI ROUTE: यह आपके HTML पेज को ब्राउज़र पर सीधे रेंडर करेगा
# =====================================================================
#  100% सही कोड:
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    # FastAPI और Jinja2 के नए वर्शन्स में context डिक्शनरी को अलग से पास करना सुरक्षित रहता है
    return templates.TemplateResponse(request=request, name="index.html")



# =====================================================================
# API ENDPOINTS
# =====================================================================
@app.post("/predict/one-to-one")
def predict_one_to_one(request: TextRequest):
    if "m11" not in models:
        raise HTTPException(status_code=500, detail="One-To-One model not loaded")
    padded = text_to_sequence(request.text, max_len=1)
    prediction = models["m11"].predict(padded, verbose=0)
    pred_idx = np.argmax(prediction, axis=-1).flatten()
    words = [idx2word[idx] for idx in pred_idx if idx2word.get(idx) != "<PAD>"]
    return {"input": request.text, "prediction": " ".join(words)}



@app.post("/predict/many-to-one")
def predict_many_to_one(request: TextRequest):
    if "mM1" not in models:
        raise HTTPException(status_code=500, detail="Many-To-One model not loaded")
        
    padded = text_to_sequence(request.text, max_len=10)
    prediction = models["mM1"].predict(padded, verbose=0)
    
    # FIXED: .item() का उपयोग करके एरे से सिंगल इंटीजर इंडेक्स बाहर निकालें
    pred_index = int(np.argmax(prediction, axis=-1).flatten()[0])
    
    word = idx2word.get(pred_index, "<UNKNOWN>")
    return {"input": request.text, "prediction": word}


@app.post("/predict/many-to-one")
def predict_many_to_one(request: TextRequest):
    if "mM1" not in models:
        raise HTTPException(status_code=500, detail="Many-To-One model not loaded")
    padded = text_to_sequence(request.text, max_len=10)
    prediction = models["mM1"].predict(padded, verbose=0)
    pred_index = np.argmax(prediction, axis=-1).flatten()
    word = idx2word.get(pred_index, "<UNKNOWN>")
    return {"input": request.text, "prediction": word}

@app.post("/predict/many-to-many")
def predict_many_to_many(request: TextRequest):
    if "mMM" not in models:
        raise HTTPException(status_code=500, detail="Many-To-Many model not loaded")
    padded = text_to_sequence(request.text, max_len=10)
    prediction = models["mMM"].predict(padded, verbose=0)
    pred_indices = np.argmax(prediction, axis=-1).flatten()
    words = [idx2word[idx] for idx in pred_indices if idx2word.get(idx) != "<PAD>"]
    return {"input": request.text, "prediction": " ".join(words)}


@app.post("/predict/combine")
async def predict_combine(text: str = Form(...), file: UploadFile = File(...)):
    if "mCombine" not in models:
        raise HTTPException(status_code=500, detail="Combined CNN_RNN model not loaded")
        
    temp_img_path = ""
    try:
        # 1. अपलोड की गई इमेज को टेम्परेरी सेव करें
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_img_path = os.path.join(temp_dir, file.filename)

     
        with open(temp_img_path, "wb") as buffer:
            buffer.write(await file.read())
            
        # 2. इनपुट डेटा मैट्रिसेस तैयार करें
        padded_text = text_to_sequence(text, max_len=10)
        processed_img = preprocess_uploaded_image(temp_img_path)
        
        # 3. प्रेडिक्शन रन करें
        prediction = models["mCombine"].predict([padded_text, processed_img], verbose=0)
        probability = float(np.squeeze(prediction))
        
        # काम होने के बाद टेम्परेरी फ़ाइल को डिलीट करें
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)
            
        # 4. डिसीजन लॉजिक
        user_details = None
        is_match = probability >= 0.5
        
        if is_match:
            csv_path = os.path.join("artifacts", "raw", "raw.csv")
            if os.path.exists(csv_path):
                df_db = pd.read_csv(csv_path)
                
                #  ROBUST STRING CLEANING FUNCTION (यह सारे कॉमा, स्पेस हटाकर सिर्फ अक्षरों को मैच करता है)
                def clean_string(s):
                    return "".join(c for c in str(s).lower() if c.isalnum())
                
                # दोनों तरफ के स्ट्रिंग्स को क्लीन करके मैच करें
                cleaned_input = clean_string(text)
                matched_rows = df_db[df_db['address'].apply(clean_string) == cleaned_input]
                
                # अगर परफेक्ट मैच न मिले, तो Substring Search (आंशिक मैच) ट्राई करें
                if matched_rows.empty:
                    matched_rows = df_db[df_db['address'].astype(str).str.lower().str.contains(text.strip().lower(), na=False)]
                
                if not matched_rows.empty:
                    row = matched_rows.iloc[0]
                    user_details = {
                        "name": str(row.get("name", "")),
                        "age": int(row.get("age", 0)),
                        "address": str(row.get("address", "")),
                        "year": int(row.get("year", 0)),
                        "email": str(row.get("email", "")),
                        "class": str(row.get("class", "")),
                        "photo_path": str(row.get("photo_path", ""))
                    }
        
        return {
            "text_input": text,
            "image_name": file.filename,
            "probability": probability,
            "classification": "Positive/Match" if is_match else "Negative/Mismatch",
            "match_found": user_details is not None,
            "details": user_details
        }
        
    except Exception as e:
        if temp_img_path and os.path.exists(temp_img_path):
            os.remove(temp_img_path)
        logger.error(f"Error in predict_combine: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
