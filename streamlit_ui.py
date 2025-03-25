import streamlit as st
import requests
import os
from dotenv import load_dotenv
import uuid
from pymongo import MongoClient
import bcrypt

load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8000")
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI, tls=True, tlsAllowInvalidCertificates=False)
db = client["cognichat_db"]
chat_collection = db["chats"]

st.title("CogniChat")

# Session State
if "user_id" not in st.session_state:
    st.session_state.user_id = None
    st.session_state.conversation_id = None

# Profile UI
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

# Chat UI
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
