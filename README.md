# RAG and Crop Price Prediction

This application integrates a Retrieval-Augmented Generation (RAG) system with a crop price prediction model using FastAPI.

## Setup

1. Clone the repository.
2. Navigate to the `farmergpt` directory.
3. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
4. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
5. Set up the `.env` file with your OpenAI API key.

6. Run the application:
    ```bash
    uvicorn backend.app.main:app --reload
    ```

7. Access the application by opening `http://localhost:8000` in your web browser.

## Features

- Upload PDFs and process them.
- Query a RAG model for information.
- Predict crop prices based on historical data.

## Folder Structure

- `/backend`: Contains the FastAPI application logic, models, and utilities.
- `/frontend`: Contains the HTML, CSS, and JavaScript files for the frontend.
- `/storage`: Holds datasets, models, and other static files.
