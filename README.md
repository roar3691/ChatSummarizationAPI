```markdown
# Chatbot

This is a chatbot application combining a FastAPI backend and Streamlit frontend, deployed on Hugging Face Spaces at (https://huggingface.co/spaces/roar3691/chatbot).

## Features
- User authentication (sign up/login).
- Chat with an AI powered by OpenRouter and Google Search.
- Conversation summarization.
- Chat history with rating functionality.

## Setup Instructions

### Prerequisites
- Python 3.10
- MongoDB instance (e.g., MongoDB Atlas)
- API keys for Google Custom Search and OpenRouter

### Local Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/roar3691/chatbot-assignment.git
   cd chatbot-assignment
   ```
2. Set environment variables in a `.env` file:
   ```
   GOOGLE_API_KEY=your_google_api_key
   SEARCH_ENGINE_ID=your_search_engine_id
   OPENROUTER_API_KEY=sk-or-v1-...
   MONGO_URI=mongodb+srv://...
   ```
3. Run the app:
   ```bash
   python app.py
   ```
4. Open `http://localhost:7860` in your browser.


## Files
- `app.py`: Main application script with FastAPI and Streamlit.
- `streamlit_ui.py`: Streamlit UI for chatting with the FastAPI backend.

## License
MIT License
```

Dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY app.py .
EXPOSE 7860
CMD ["bash", "-c", "uvicorn app:app --host 0.0.0.0 --port 8000 & streamlit run app.py --server.port 7860 --server.address 0.0.0.0"]
```

---
