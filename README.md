

# Lightning-Fast RAG App with LangChain, Groq, and Streamlit

This project demonstrates a powerful and highly performant Retrieval Augmented Generation (RAG) application built using LangChain for orchestration, Groq for rapid LLM inference, and Streamlit for an interactive web interface. It's designed to quickly answer user questions by retrieving relevant information from a document corpus and leveraging the speed of Groq's Language Processing Unit (LPU).

## Features

* **Rapid LLM Inference:** Utilizes Groq's LPU for incredibly fast response times, providing a seamless user experience.
* **Efficient Document Processing:** Employs LangChain's `WebBaseLoader` to load documents and `RecursiveCharacterTextSplitter` for optimal chunking.
* **Persistent Vector Store:** Uses FAISS to store and retrieve document embeddings, allowing for quick loading of pre-processed data.
* **Contextual Question Answering:** Leverages LangChain's RAG chain to ensure LLM responses are grounded in the provided document context, minimizing hallucinations.
* **Interactive Web UI:** Built with Streamlit for easy interaction, allowing users to input questions and view responses directly in a web browser.
* **Performance Measurement:** Includes a simple time measurement to showcase the speed of Groq's inference.

## Getting Started

### Prerequisites

* Python 3.8+
* A Groq API Key (set as an environment variable `GROQ_API_KEY`)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    (You'll need to create a `requirements.txt` file with: `streamlit`, `langchain-groq`, `langchain-community`, `langchain`, `faiss-cpu`, `ollama`, `python-dotenv`)
4.  **Set up your Groq API Key:**
    Create a `.env` file in the root directory of your project and add your Groq API key:
    ```
    GROQ_API_KEY="your_groq_api_key_here"
    ```

### Running the Application

1.  **Start the Streamlit app:**
    ```bash
    streamlit run your_app_file_name.py # Replace with your Python script name
    ```
2.  **Access the application:**
    Your browser will automatically open to `http://localhost:8501` (or a similar address).

### Usage

1.  Click the "Document Embeddings" button to load and process the initial documents (or load the existing FAISS index).
2.  Once embeddings are ready, enter your questions in the text input field.
3.  The application will retrieve relevant information and provide an answer using the Groq LLM.


---
