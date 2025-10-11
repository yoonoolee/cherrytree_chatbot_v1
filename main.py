from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import requests
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Allow requests from your Squarespace site
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store conversations in memory
conversations = {}

class Message(BaseModel):
    text: str
    session_id: str


@app.post("/chat")
async def chat(msg: Message):
    """Receive message and return Gemini response"""
    
    session_id = msg.session_id
    user_text = msg.text
    
    # Initialize session if new
    if session_id not in conversations:
        conversations[session_id] = []
    
    # Add user message
    conversations[session_id].append({
        "role": "user",
        "parts": [{"text": user_text}]
    })
    
    # Call Gemini
    try:
        payload = {"contents": conversations[session_id]}
        params = {"key": os.getenv("GEMINI_API_KEY")}
        
        response = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
            json=payload,
            params=params,
            timeout=30
        )
        response.raise_for_status()
        
        data = response.json()
        bot_message = data["candidates"][0]["content"]["parts"][0]["text"]
        
        # Add bot response
        conversations[session_id].append({
            "role": "model",
            "parts": [{"text": bot_message}]
        })
        
        return {"response": bot_message, "session_id": session_id}
    
    except Exception as e:
        return {"error": str(e), "response": "Sorry, something went wrong."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)