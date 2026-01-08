import os
import re
import pickle
import nltk
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import PyPDF2
import docx
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
from dotenv import load_dotenv
load_dotenv()

# Download required NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    pass

# Initialize FastAPI instance
app = FastAPI(title="Document Classification API")

_cors_origins_raw = os.getenv("CORS_ORIGINS", "*").strip()
if not _cors_origins_raw or _cors_origins_raw == "*":
    CORS_ORIGINS = ["*"]
else:
    CORS_ORIGINS = [origin.strip() for origin in _cors_origins_raw.split(",") if origin.strip()]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
MODEL_PATH = "models/svm_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"
ENCODER_PATH = "models/label_encoder.pkl"

# Global variables for model storage
model = None
vectorizer = None
label_encoder = None

class ProcessingError(Exception):
    pass

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def clean_text(self, text):
        # 1. Lowercase
        text = text.lower()
        # 2. Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        # 3. Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        # 4. Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        # 5. Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # 6. Tokenize
        tokens = text.split()
        # 7. Remove stop words and short words
        tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]
        # 8. Lemmatize
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        
        return ' '.join(tokens)

preprocessor = TextPreprocessor()

def load_models():
    global model, vectorizer, label_encoder
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(VECTORIZER_PATH, 'rb') as f:
            vectorizer = pickle.load(f)
        with open(ENCODER_PATH, 'rb') as f:
            label_encoder = pickle.load(f)
        print("Models loaded successfully")
    except Exception as e:
        print(f"Error loading models: {e}")

@app.on_event("startup")
async def startup_event():
    load_models()

# Extraction Functions
async def extract_from_pdf(file_content: bytes) -> str:
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        raise ProcessingError(f"Error extracting PDF: {str(e)}")

async def extract_from_docx(file_content: bytes) -> str:
    try:
        doc = docx.Document(io.BytesIO(file_content))
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        raise ProcessingError(f"Error extracting DOCX: {str(e)}")

async def extract_from_txt(file_content: bytes) -> str:
    try:
        return file_content.decode('utf-8')
    except UnicodeDecodeError:
        try:
            return file_content.decode('latin-1')
        except:
            raise ProcessingError("Error decoding text file")

class ClassificationResponse(BaseModel):
    filename: str
    prediction: str
    confidence: float
    message: str

@app.get("/")
async def root():
    return {"status": "healthy", "service": "Document Classification API"}

@app.post("/classify", response_model=ClassificationResponse)
async def classify_file(file: UploadFile = File(...)):
    if not model or not vectorizer or not label_encoder:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    content = await file.read()
    filename = file.filename.lower()
    text = ""
    extracted = False

    # Logic to try all parsers regardless of extension
    parsers = [
        (extract_from_pdf, "pdf"),
        (extract_from_docx, "docx"),
        (extract_from_txt, "txt")
    ]

    # Prioritize based on extension if known
    ext = filename.split('.')[-1]
    if ext == 'pdf':
        parsers.insert(0, parsers.pop(0))
    elif ext in ['docx', 'doc']:
        parsers.insert(0, parsers.pop(1))
    elif ext == 'txt':
        parsers.insert(0, parsers.pop(2))

    errors = []
    for parser, ptype in parsers:
        try:
            text = await parser(content)
            if len(text.strip()) > 50: # Check if enough text was extracted
                extracted = True
                break
        except Exception as e:
            errors.append(f"{ptype}: {str(e)}")
            continue

    if not extracted or not text.strip():
        raise HTTPException(
            status_code=400, 
            detail=f"Could not extract sufficient text from file. Errors: {'; '.join(errors)}"
        )

    # Preprocess
    cleaned_text = preprocessor.clean_text(text)
    
    if not cleaned_text:
        raise HTTPException(status_code=400, detail="Text cleaning resulted in empty string")

    # Vectorize
    features = vectorizer.transform([cleaned_text])

    # Predict
    prediction_idx = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    prediction_label = label_encoder.inverse_transform([prediction_idx])[0]
    confidence = float(probabilities[prediction_idx])

    return ClassificationResponse(
        filename=file.filename,
        prediction=prediction_label,
        confidence=confidence,
        message="Classification successful"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
