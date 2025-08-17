# RAG Document Q&A System

A production-ready Retrieval-Augmented Generation (RAG) system for document question-answering with clean web interface.

## 🚀 Features

- **Document Upload**: Support for PDF, TXT, and DOCX files
- **Intelligent Q&A**: Ask questions and get answers with source citations
- **Vector Search**: Powered by ChromaDB for fast semantic search
- **Clean Interface**: Streamlit-based web UI
- **Production Ready**: Docker support, comprehensive logging, monitoring
- **RESTful API**: FastAPI-based backend with automatic documentation

## 🛠 Tech Stack

- **Backend**: Python, FastAPI, SQLAlchemy
- **LLM**: OpenAI GPT-3.5/4 with embeddings
- **Vector DB**: ChromaDB
- **Database**: PostgreSQL
- **Frontend**: Streamlit
- **Containerization**: Docker & Docker Compose
- **Testing**: Pytest with async support

## 📋 Prerequisites

- Python 3.9+
- PostgreSQL 12+
- OpenAI API key
- Docker (optional, for containerized deployment)

## 🔧 Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/rag-document-qa.git
cd rag-document-qa