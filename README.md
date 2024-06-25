# Farmergpt 

This application integrates a Retrieval-Augmented Generation (RAG) system with a crop recommendationa and price prediction model using FastAPI.

## Setup

1. Clone the repository.
2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Set up the `.env` file with your OpenAI API key.
    ```bash
    OPENAI_API_KEY=your_openai_api_key
    ```

6. Run the application:
    ```bash
    uvicorn main:app --reload
    ```

7. Access the application by opening `http://localhost:8000` in your web browser.

## Features

- Upload PDFs and process them.
- Query a RAG model for information.
- Predict crop prices based on historical data.
- predict crop recommendation on historical data

