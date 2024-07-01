import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import logging
import joblib

import logging
import traceback
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
import sqlite3

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader, PyPDFDirectoryLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import StuffDocumentsChain, LLMChain
from langchain.document_loaders import PyPDFLoader
from fastapi.templating import Jinja2Templates

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OpenAI API key not found. Make sure to set it in the environment variables.")

# FastAPI app setup
app = FastAPI()

# CORS setup to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Define the path to the persisted ChromaDB
persist_directory = './chromadb_storage/db'
embedding_model = OpenAIEmbeddings()

# Load models and scaler for crop prediction
crop_prediction_model = joblib.load('models/crop_prediction_model.pkl')
scaler = joblib.load('scalers/crop_prediction_scaler.pkl')
mlb = joblib.load('scalers/mlb.pkl')

# Load dataset for crop prediction districts
file_path = 'datasets/Sri_Lankan_Soil_Quality_Data.csv'
data = pd.read_csv(file_path)
districts = data['district'].unique()

# Create the folder if it doesn't exist
os.makedirs('chat_history_db', exist_ok=True)

# Database URL
DATABASE_URL = "sqlite:///./chat_history_db/conversations.db"

# Connect to the database
conn = sqlite3.connect('./chat_history_db/conversations.db', check_same_thread=False)
cursor = conn.cursor()

# Create conversations table if it doesn't exist
cursor.execute("""
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_query TEXT,
    answer TEXT,
    sources TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")

# Commit the changes
conn.commit()

class DistrictRequest(BaseModel):
    district: str

class Query(BaseModel):
    question: str

def get_data_for_district(district):
    filtered_df = data[data['district'] == district]
    if filtered_df.empty:
        raise ValueError("District not found in the dataset")
    n = filtered_df['N'].values[0]
    p = filtered_df['P'].values[0]
    k = filtered_df['K'].values[0]
    temperature = filtered_df['temperature'].values[0]
    humidity = filtered_df['humidity'].values[0]
    rainfall = filtered_df['rainfall'].values[0]
    ph = filtered_df['ph'].values[0]
    input_data = np.array([[n, p, k, temperature, humidity, ph , rainfall]])
    return input_data

def get_top_3_labels(probs, mlb):
    top_3_indices = np.argsort(probs, axis=1)[:, -3:]
    top_3_labels = mlb.classes_[top_3_indices[0]]
    return top_3_labels

@app.get("/districts")
def read_districts():
    return {"districts": list(districts)}

@app.post("/predict_crop_recommendations")
def predict_district(request: DistrictRequest):
    district = request.district
    if district not in districts:
        raise HTTPException(status_code=404, detail="District not found")
    try:
        input_data = get_data_for_district(district)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    user_input_scaled = scaler.transform(input_data)
    logistic_probs = crop_prediction_model.predict_proba(user_input_scaled)
    logistic_labels = get_top_3_labels(logistic_probs, mlb)
    return {"district": district, "recommendations": logistic_labels.tolist()}

# Function to load and split documents
def load_and_split_documents(directory):
    #loader = DirectoryLoader(directory, glob="*.txt", loader_cls=TextLoader)
    loader = DirectoryLoader(directory, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

def initialize_chromadb(directory, persist_directory):
    try:
        logger.info(f"Loading and splitting documents from directory: {directory}")
        texts = load_and_split_documents(directory)
        logger.info(f"Loaded and split {len(texts)} documents")
        
        logger.info(f"Initializing ChromaDB with persist directory: {persist_directory}")
        vectordb = Chroma.from_documents(documents=texts,
                                         embedding=embedding_model,
                                         persist_directory=persist_directory)
        vectordb.persist()
        logger.info("ChromaDB initialized and persisted successfully")
        return vectordb
    except Exception as e:
        logger.error(f"Error initializing ChromaDB: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Failed to initialize ChromaDB")

def update_chromadb(texts, persist_directory, embedding_model):
    try:
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
        vectordb.add_documents(documents=texts)
        vectordb.persist()
        return vectordb
    except Exception as e:
        logger.error(f"Error updating ChromaDB: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Failed to update ChromaDB")


# Check if the persist directory exists and initialize or load ChromaDB
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)
    vectordb = initialize_chromadb('./new_papers/', persist_directory)
else:
    try:
        vectordb = Chroma(persist_directory=persist_directory,
                          embedding_function=embedding_model)
    except Exception as e:
        logger.error(f"Error loading ChromaDB: {e}")
        logger.error(traceback.format_exc())
        vectordb = initialize_chromadb('./new_papers/', persist_directory)

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
    
    
@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
        
        content = await file.read()
        file_path = f'./new_papers/{file.filename}'
        with open(file_path, 'wb') as f:
            f.write(content)

        texts = load_and_split_documents('./new_papers/')
        vectordb = update_chromadb(texts, persist_directory, embedding_model)
        return {"filename": file.filename, "status": "Processed"}
    except Exception as e:
        logger.error(f"Error uploading or processing PDF: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Failed to upload or process PDF")
    
@app.post("/query/")
async def query_rag(query: Query):
    try:
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model_name='gpt-3.5-turbo'),
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        response = qa_chain(query.question)

        cursor.execute("""
        INSERT INTO conversations (user_query, answer, sources)
        VALUES (?, ?, ?)
        """, (query.question, response['result'], ', '.join([doc.metadata['source'] for doc in response['source_documents']])))
        conn.commit()

        return {"answer": response['result'], "sources": [doc.metadata['source'] for doc in response['source_documents']]}
    except Exception as e:
        logger.error(f"Error querying RAG: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Failed to process the query")
    
@app.get("/conversations/", response_model=List[dict])
async def get_conversations():
    try:
        cursor.execute("SELECT * FROM conversations")
        rows = cursor.fetchall()
        conversations = [
            {"id": row[0], "user_query": row[1], "answer": row[2], "sources": row[3], "timestamp": row[4]}
            for row in rows
        ]
        return conversations
    except Exception as e:
        logger.error(f"Error retrieving conversations: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Failed to retrieve conversations")

@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: int):
    try:
        cursor.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        conn.commit()
        return {"message": "Conversation deleted"}
    except Exception as e:
        logger.error(f"Error deleting conversation: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Failed to delete conversation")

# Define file paths for models, scalers, and datasets
data_paths = {
    "kidneybeans": 'datasets/kidneybeans_data.csv',
    "banana": 'datasets/banana_data.csv',
    "chickpeas": 'datasets/chickpea_data.csv',
    "coconut": 'datasets/coconut_data.csv',
    "papaya": 'datasets/papaya_data.csv',
    "rice": 'datasets/rice_data.csv',
    "turmeric": 'datasets/turmeric_data.csv'
}

model_paths = {
    "kidneybeans": 'models/kidneybeans_gru_model.h5',
    "banana": 'models/banana_gru_model.h5',
    "chickpeas": 'models/chickpea_gru_model.h5',
    "coconut": 'models/coconut_gru_model.h5',
    "papaya": 'models/papaya_gru_model.h5',
    "rice": 'models/rice_gru_model.h5',
    "turmeric": 'models/turmeric_gru_model.h5'
}

scaler_paths = {
    "kidneybeans": 'scalers/kidneybeans_scaler.pkl',
    "banana": 'scalers/banana_scaler.pkl',
    "chickpeas": 'scalers/chickpea_scaler.pkl',
    "coconut": 'scalers/coconut_scaler.pkl',
    "papaya": 'scalers/papaya_scaler.pkl',
    "rice": 'scalers/rice_scaler.pkl',
    "turmeric": 'scalers/turmeric_scaler.pkl'
}

class PricePredictionRequest(BaseModel):
    crop: str  # Added crop selection
    year: int
    month: int
    week: int

def load_resources(crop):
    if crop not in data_paths or crop not in model_paths or crop not in scaler_paths:
        raise ValueError(f"Crop '{crop}' not supported")

    # Load dataset
    df = pd.read_csv(data_paths[crop])
    
    # Convert Year, Month, Week into a single datetime index
    df['date'] = pd.to_datetime(df[['Year', 'Month']].assign(Day=1)) + pd.to_timedelta((df['Week'] - 1) * 7, unit='d')
    df = df.set_index('date')
    df = df.sort_index()
    
    # Load the GRU model
    model = tf.keras.models.load_model(model_paths[crop])
    
    # Load the scaler
    with open(scaler_paths[crop], 'rb') as f:
        scaler = pickle.load(f)
    
    return df, model, scaler

def predict_price(crop, year, month, week):
    df, model, scaler = load_resources(crop)
    
    future_date = pd.to_datetime({'year': [year], 'month': [month], 'day': [1]}) + pd.to_timedelta((week - 1) * 7, unit='d')
    future_date = future_date[0]

    latest_date = df.index[-1]
    time_steps = 4

    if future_date <= latest_date:
        start_date = future_date - pd.Timedelta(weeks=time_steps)
        previous_weeks = df.loc[start_date:future_date - pd.Timedelta(days=1), 'Price'].values
        if len(previous_weeks) < time_steps:
            raise ValueError(f"Not enough data to predict price for {year}-{month}-{week}. Need at least {time_steps} weeks of data before the date.")
        
        previous_weeks_scaled = scaler.transform(previous_weeks.reshape(-1, 1))
        X_future = previous_weeks_scaled.reshape((1, time_steps, 1))
        future_prediction_scaled = model.predict(X_future)
        future_prediction = scaler.inverse_transform(future_prediction_scaled)
        return float(future_prediction[0, 0])  # Convert to native float

    predictions = []
    latest_data = df.loc[latest_date - pd.Timedelta(weeks=time_steps - 1):latest_date, 'Price'].values
    if len(latest_data) < time_steps:
        latest_data = np.pad(latest_data, (time_steps - len(latest_data), 0), 'constant', constant_values=(0,))
    latest_data_scaled = scaler.transform(latest_data.reshape(-1, 1)).reshape((1, time_steps, 1))

    current_date = latest_date
    while current_date < future_date:
        future_prediction_scaled = model.predict(latest_data_scaled)
        future_prediction = scaler.inverse_transform(future_prediction_scaled)
        predictions.append(future_prediction[0, 0])

        new_value_scaled = future_prediction_scaled.flatten()[0]
        latest_data_scaled = np.append(latest_data_scaled[0][1:], [[new_value_scaled]], axis=0).reshape((1, time_steps, 1))
        current_date += pd.Timedelta(weeks=1)

    return float(predictions[-1])  # Convert to native float

@app.post("/predict_price/")
def get_price_prediction(request: PricePredictionRequest):
    try:
        price = predict_price(request.crop, request.year, request.month, request.week)
        return {"predicted_price": price}
    except ValueError as e:
        logging.error(f"Value error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logging.exception("An unexpected error occurred")
        raise HTTPException(status_code=500, detail="Internal Server Error. Check logs for details.")

# Run the app using Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

logging.basicConfig(level=logging.DEBUG)


@app.exception_handler(Exception)
async def all_exception_handler(request, exc):
    logging.exception("An error occurred")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error. Please check the logs for more details."},
    )

