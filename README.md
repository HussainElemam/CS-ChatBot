# CS ChatBot

A full-stack Retrieval-Augmented Generation (RAG) chatbot for answering questions from computer science course materials. The project combines a React chat interface with a Flask API, LangChain document retrieval, a local Chroma vector store, and Ollama-hosted language and embedding models.

## Purpose

CS ChatBot was built to make course content easier to search and understand through natural conversation. Instead of manually looking through lecture files, a user can ask a question and receive an answer grounded in the indexed course documents.

The project highlights full-stack development, LLM application design, local AI tooling, vector search, and conversational UI implementation.

## Features

- Chat interface built with React, TypeScript, and Vite
- Flask backend API for handling chat requests
- RAG pipeline using LangChain and Chroma
- Local model inference through Ollama
- Document ingestion for PDF, DOCX, and PPTX course materials
- Conversation history support for follow-up questions
- Automatic text chunking and vector-store generation

## How It Works

1. Course documents are loaded from `backend/data/course_materials`.
2. The backend splits the documents into searchable chunks.
3. Chunks are embedded with `nomic-embed-text` and stored in Chroma.
4. The React frontend sends user messages and recent chat history to the Flask API.
5. The backend reformulates follow-up questions, retrieves relevant course context, and asks the Ollama LLM to answer using that context.

## Tech Stack

- Frontend: React, TypeScript, Vite, CSS
- Backend: Python, Flask, Flask-CORS
- AI/RAG: LangChain, Ollama, Chroma
- Models: `qwen2.5:7b` for generation and `nomic-embed-text` for embeddings

## Project Structure

```text
CS-ChatBot/
|-- backend/
|   |-- app.py
|   |-- requirements.txt
|   |-- data/
|   |   `-- course_materials/
|   `-- src/
|       |-- create_database.py
|       `-- retrieve_data.py
|-- frontend/
|   |-- package.json
|   `-- src/
|       `-- App.tsx
`-- README.md
```

## Running Locally

Prerequisites:

- Python 3.10+
- Node.js and npm
- Ollama installed and running locally

Pull the required Ollama models:

```bash
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

Set up the backend:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/create_database.py
python app.py
```

Set up the frontend in a second terminal:

```bash
cd frontend
npm install
npm run dev
```

The frontend runs with Vite, and the backend serves chat responses from `http://localhost:5001/api/chat`.

## What I Built

- Implemented a document ingestion pipeline for course files
- Built a local vector database workflow for semantic retrieval
- Connected a conversational React UI to a Flask backend
- Added chat history handling so follow-up questions can be interpreted in context
- Used local Ollama models to keep the application independent from hosted AI APIs
