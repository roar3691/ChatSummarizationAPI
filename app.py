import os
import asyncio
import aiohttp
import json
import logging
import re
import subprocess
import sys
import threading
import uuid

# Dynamic Package Installation
def install_package(package):
    try:
        __import__(package.split("==")[0].replace("-", "_"))
    except ImportError:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package}: {e}")
            sys.exit(1)

required_packages = [
    "fastapi==0.103.1",
    "uvicorn==0.23.2",
    "pymongo==4.5.0",
    "python-dotenv==1.0.0",
    "aiohttp==3.8.5",
    "bcrypt==4.0.1",
    "google-api-python-client==2.100.0",
    "streamlit==1.27.0",
    "requests==2.31.0"
]

for package in required_packages:
    install_package(package)

# Import after installation
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pymongo import MongoClient, ASCENDING
from datetime import datetime
from googleapiclient.discovery import build
from typing import List, Optional, Dict
from pydantic import BaseModel
from bson import ObjectId
import uvicorn
import bcrypt

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")
API_URL = "http://localhost:8000"

# Logging setup
logging.basicConfig(level=logging.INFO, filename="chatbot.log", format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# MongoDB Setup
client = MongoClient(MONGO_URI, tls=True, tlsAllowInvalidCertificates=False)
db = client["cognichat_db"]
chat_collection = db["chats"]
profile_collection = db["profiles"]
chat_collection.create_index([("user_id", ASCENDING), ("conversation_id", ASCENDING), ("timestamp", ASCENDING)])

# FastAPI Setup
app = FastAPI(title="CogniChat API")

# Pydantic Models
class ChatMessage(BaseModel):
    user_id: str
    message: str
    conversation_id: Optional[str] = None

class SummaryRequest(BaseModel):
    conversation_id: str

class UserProfile(BaseModel):
    username: str
    password: str

# Helper Functions
def heuristic_multi_scale_attention(query: str) -> float:
    words = query.lower().split()
    length = len(words)
    short_scale = min(1.0, length / 5)
    specific_keywords = {"what", "how", "why", "who", "where", "when", "explain", "describe"}
    mid_scale = sum(1 for word in words if word in specific_keywords) / max(1, length)
    long_scale = 1.0 if "?" in query or len(re.findall(r"\w+", query)) > 3 else 0.5
    return min(max(0.3 * short_scale + 0.4 * mid_scale + 0.3 * long_scale, 0.1), 1.0)

async def perform_google_search(query: str) -> str:
    try:
        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        res = await asyncio.to_thread(service.cse().list(q=query, cx=SEARCH_ENGINE_ID, num=3).execute)
        return "\n".join([f"- {item['title']}: {item['snippet']}" for item in res.get("items", [])]) or "No results."
    except Exception as e:
        logger.error(f"Google Search error: {e}")
        return f"Search error: {e}"

async def query_llm(prompt: str, user_id: str) -> str:
    preferences = profile_collection.find_one({"user_id": user_id}, {"preferences": 1})
    prefs = preferences.get("preferences", {"tone": "formal", "detail_level": "medium", "greeting": "Hello"}) if preferences else {"tone": "formal", "detail_level": "medium", "greeting": "Hello"}
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
                json={"model": "deepseek/deepseek-chat-v3-0324:free", "messages": [{"role": "user", "content": prompt}], "temperature": 0.5}
            ) as response:
                result = await response.json()
                if "choices" not in result:
                    raise HTTPException(status_code=500, detail="Invalid LLM response")
                return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"LLM error: {e}")
            raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")

# FastAPI Endpoints
@app.post("/chats", status_code=201)
async def store_chat(chat: ChatMessage):
    conversation_id = chat.conversation_id or str(uuid.uuid4())
    focus_score = heuristic_multi_scale_attention(chat.message)
    prompt = f"""
    **User Query**: "{chat.message}"
    **Focus Score**: {focus_score:.2f}
    **Instructions**: Respond in a formal tone with medium detail. Use Google Search if relevant.
    """
    response = await query_llm(prompt, chat.user_id)
    google_results = await perform_google_search(chat.message) if focus_score > 0.7 else "N/A"
    chat_entry = {
        "user_id": chat.user_id,
        "conversation_id": conversation_id,
        "user_message": chat.message,
        "ai_response": response,
        "google_results": google_results,
        "timestamp": datetime.utcnow().timestamp(),
        "rating": None,
        "rating_comment": None
    }
    chat_collection.insert_one(chat_entry)
    return {"conversation_id": conversation_id, "response": response}

@app.get("/chats/{conversation_id}")
async def get_chat(conversation_id: str):
    chat = chat_collection.find_one({"conversation_id": conversation_id})
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    chat["_id"] = str(chat["_id"])
    return chat

@app.post("/chats/summarize")
async def summarize_chat(request: SummaryRequest):
    chats = list(chat_collection.find({"conversation_id": request.conversation_id}).sort("timestamp", ASCENDING))
    if not chats:
        raise HTTPException(status_code=404, detail="Conversation not found")
    history = "\n".join([f"User: {c['user_message']}\nAI: {c['ai_response']}" for c in chats])
    prompt = f"""
    Summarize this conversation and provide insights (sentiment, keywords):
    {history}
    """
    summary = await query_llm(prompt, chats[0]["user_id"])
    return {"conversation_id": request.conversation_id, "summary": summary}

@app.get("/users/{user_id}/chats")
async def get_user_chats(user_id: str, page: int = Query(1, ge=1), limit: int = Query(10, ge=1, le=100)):
    try:
        skip = (page - 1) * limit
        chats = list(chat_collection.find({"user_id": user_id}).sort("timestamp", -1).skip(skip).limit(limit))
        total = chat_collection.count_documents({"user_id": user_id})
        for chat in chats:
            chat["_id"] = str(chat["_id"])
        response = {"chats": chats, "page": page, "limit": limit, "total": total}
        logger.info(f"GET /users/{user_id}/chats response: {json.dumps(response)[:200]}")
        return response
    except Exception as e:
        logger.error(f"Error in get_user_chats: {e}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.delete("/chats/{conversation_id}", status_code=204)
async def delete_chat(conversation_id: str):
    result = chat_collection.delete_many({"conversation_id": conversation_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Chat not found")
    return None

@app.post("/users", status_code=201)
async def create_user(profile: UserProfile):
    if profile_collection.find_one({"username": profile.username}):
        raise HTTPException(status_code=400, detail="Username exists")
    user_id = str(uuid.uuid4())
    hashed_password = bcrypt.hashpw(profile.password.encode(), bcrypt.gensalt())
    profile_collection.insert_one({
        "user_id": user_id,
        "username": profile.username,
        "password": hashed_password,
        "preferences": {"tone": "formal", "detail_level": "medium", "greeting": "Hello"}
    })
    return {"user_id": user_id}

@app.get("/users/{user_id}/profile")
async def get_user_profile(user_id: str):
    profile = profile_collection.find_one({"user_id": user_id})
    if not profile:
        raise HTTPException(status_code=404, detail="User not found")
    profile["_id"] = str(profile["_id"])
    return profile

@app.post("/login")
async def login(profile: UserProfile):
    user = profile_collection.find_one({"username": profile.username})
    if not user or not bcrypt.checkpw(profile.password.encode(), user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"user_id": user["user_id"]}

# Streamlit UI (runs only if script is main entry point)
def run_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Write Dockerfile
dockerfile_content = """
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 7860

CMD ["bash", "-c", "uvicorn app:app --host 0.0.0.0 --port 8000 & streamlit run app.py --server.port 7860 --server.address 0.0.0.0"]
"""
with open("Dockerfile", "w") as f:
    f.write(dockerfile_content)

if __name__ == "__main__":
    # Start FastAPI in a background thread
    fastapi_thread = threading.Thread(target=run_fastapi)
    fastapi_thread.daemon = True
    fastapi_thread.start()

    # Streamlit UI (runs in main thread)
    import streamlit as st
    import requests

    st.title("CogniChat")

    if "user_id" not in st.session_state:
        st.session_state.user_id = None
        st.session_state.conversation_id = None

    def profile_ui():
        if not st.session_state.user_id:
            action = st.sidebar.radio("Action", ["Login", "Sign Up"])
            username = st.sidebar.text_input("Username")
            password = st.sidebar.text_input("Password", type="password")
            if st.sidebar.button(action):
                if action == "Sign Up":
                    resp = requests.post(f"{API_URL}/users", json={"username": username, "password": password})
                    if resp.status_code == 201:
                        st.session_state.user_id = resp.json()["user_id"]
                        st.sidebar.success("Signed up!")
                        st.rerun()
                    else:
                        st.sidebar.error(resp.json()["detail"])
                else:
                    resp = requests.post(f"{API_URL}/login", json={"username": username, "password": password})
                    if resp.status_code == 200:
                        st.session_state.user_id = resp.json()["user_id"]
                        st.sidebar.success("Logged in!")
                        st.rerun()
                    else:
                        st.sidebar.error(resp.json()["detail"])
        else:
            st.sidebar.write(f"User: {st.session_state.user_id[:8]}")
            if st.sidebar.button("Logout"):
                st.session_state.user_id = None
                st.session_state.conversation_id = None
                st.rerun()

    def chat_ui():
        if not st.session_state.user_id:
            st.warning("Please log in or sign up.")
            return

        query = st.chat_input("Type your message...")
        if query:
            if not st.session_state.conversation_id:
                st.session_state.conversation_id = str(uuid.uuid4())
            resp = requests.post(f"{API_URL}/chats", json={
                "user_id": st.session_state.user_id,
                "message": query,
                "conversation_id": st.session_state.conversation_id
            })
            if resp.status_code == 201:
                st.session_state.conversation_id = resp.json()["conversation_id"]
                st.write(f"AI: {resp.json()['response']}")
            else:
                st.error(f"Failed to send message: {resp.status_code} - {resp.text}")

        if st.button("Summarize Conversation"):
            if st.session_state.conversation_id:
                resp = requests.post(f"{API_URL}/chats/summarize", json={"conversation_id": st.session_state.conversation_id})
                if resp.status_code == 200:
                    st.write(f"Summary: {resp.json()['summary']}")
                else:
                    st.error(f"Failed to summarize: {resp.status_code} - {resp.text}")

        if st.session_state.conversation_id:
            resp = requests.get(f"{API_URL}/users/{st.session_state.user_id}/chats")
            if resp.status_code == 200:
                try:
                    chats = resp.json()["chats"]
                    for chat in reversed(chats):
                        with st.chat_message("user"):
                            st.write(chat["user_message"])
                        with st.chat_message("ai"):
                            st.write(chat["ai_response"])
                            rating = st.slider(f"Rate", 1, 5, chat.get("rating", 3), key=f"rate_{chat['timestamp']}")
                            if st.button("Submit Rating", key=f"submit_{chat['timestamp']}"):
                                chat_collection.update_one(
                                    {"conversation_id": chat["conversation_id"], "timestamp": chat["timestamp"]},
                                    {"$set": {"rating": rating}}
                                )
                                st.success(f"Rating {rating} submitted!")
                except requests.exceptions.JSONDecodeError:
                    st.error(f"Invalid response from API: {resp.status_code} - {resp.text}")
            else:
                st.error(f"Failed to fetch chats: {resp.status_code} - {resp.text}")

    profile_ui()
    chat_ui()
