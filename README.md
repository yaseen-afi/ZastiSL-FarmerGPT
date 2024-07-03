# FarmerGPT

You can access the application using this URL: [FarmerGPT](http://34.70.241.75/).

This application provides crop recommendations, price predictions, and a chatbot using Retrieval-Augmented Generation (RAG) architecture to provide tailored queries for farmers in Sri Lanka integrated using FastAPI.

## Setup

1. Clone the repository.
    ```bash
    git clone https://github.com/yourusername/farmergpt.git
    cd farmergpt
    ```

2. Create a virtual environment and activate it:

    For Linux/macOS:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

    For Windows:
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Set up the `.env` file with your OpenAI API key:
    ```bash
    OPENAI_API_KEY=your_openai_api_key
    ```

5. Run the application:
    ```bash
    uvicorn main:app --reload
    ```

6. Access the application by opening `http://localhost:8000` in your web browser.

## Features

- Upload PDFs and process them.
- Query a RAG model for information.
- Predict crop prices based on historical data.
- Predict crop recommendation based on historical data.
- Provides crop recommendations, price predictions, and a chatbot using RAG architecture to provide tailored queries for farmers in Sri Lanka.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
