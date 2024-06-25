import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import logging

# FastAPI app setup
app = FastAPI()

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
