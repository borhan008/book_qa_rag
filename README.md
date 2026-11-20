# Bengali Book QA RAG

<video controls width="720">
  <source src="./video_preview.mp4" type="video/mp4">
  Your browser does not support embedded video. [Download the video preview](./video_preview.mp4).
</video>

If the video player is not available in your Markdown viewer, [download the video preview](./video_preview.mp4).

A Bengali question-answering application for the book **মেজদিদি**. The project uses a Streamlit chat interface and a FastAPI retrieval-augmented generation (RAG) backend.

## Features

- Ask questions about the book in Bengali.
- Retrieve relevant passages with Bengali sentence embeddings.
- Generate answers with an OpenRouter-hosted language model.
- Include the source chapter in the generated response.
- Provide both a browser chat interface and HTTP API endpoints.

## How it works

1. When the FastAPI application starts, it downloads the configured chapters from Bengali Wikisource.
2. The chapter text is cleaned and split into overlapping chunks of 500 characters with 100 characters of overlap.
3. Each chunk is converted into an embedding using `l3cube-pune/bengali-sentence-similarity-sbert`.
4. The embeddings are indexed in a FAISS vector store.
5. A question sent from the Streamlit interface is compared with the indexed chunks, and the three most similar chunks are selected.
6. Those chunks are passed as context to the OpenRouter model. The prompt instructs the model to answer in Bengali, mention the chapter, and say when the answer is not present in the context.
7. The Streamlit app displays the response as part of the chat history.

The index is built in memory during backend startup, so the initial launch can take longer and requires internet access.

## Project structure

```text
.
├── app.py              # Streamlit chat interface
├── main.py             # FastAPI API and RAG pipeline
├── requirements.txt    # Python dependencies
└── video_preview.mp4   # Application preview video
```

## Requirements

- Python 3.10 or newer
- Internet access for Wikisource, Hugging Face model downloads, and OpenRouter
- An OpenRouter API key

## Installation

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Configuration

Before starting the backend, set the OpenRouter API key in the `ChatOpenRouter` configuration in `main.py`:

```python
llm = ChatOpenRouter(
    model="nvidia/nemotron-3.5-lightning:free",
    temperature=0.7,
    api_key="YOUR_OPENROUTER_API_KEY",
)
```

Do not commit a real API key to the repository. For production use, load it from a secret or environment variable instead.

## Running the application

Start the FastAPI backend in one terminal:

```powershell
uvicorn main:app --reload
```

Then start the Streamlit frontend in a second terminal:

```powershell
streamlit run app.py
```

Open the URL printed by Streamlit, usually `http://localhost:8501`.

The frontend expects the backend to be available at `http://localhost:8000`. If the backend is unavailable, the chat displays a connection error.

## API endpoints

### Health check

```http
GET /
```

Returns:

```json
{"message": "FastAPI is running!"}
```

### Ask a book question

```http
GET /ask?question=মেজদিদির প্রধান চরিত্র কে?
```

Returns the generated answer:

```json
{"response": "বাংলায় তৈরি উত্তর..."}
```

### Echo endpoint

```http
GET /chat?message=hello
```

Returns the supplied message. This endpoint is included as a simple API example and is not part of the RAG flow.

## Notes

- The application fetches source pages every time the backend starts.
- The FAISS index is not persisted to disk; it is rebuilt on each startup.
- Answers are constrained by the retrieved context, but generated responses should still be reviewed for accuracy.
