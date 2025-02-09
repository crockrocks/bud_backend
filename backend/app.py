import os
import json
import logging
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import Dict, List

from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from bson import ObjectId
import firebase_admin
from firebase_admin import credentials, auth
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Logging setup
logging.basicConfig(level=logging.INFO)

# Initialize Firebase Admin
firebase_cred_path = os.getenv("FIREBASE_CREDENTIALS")
if not firebase_cred_path or not os.path.exists(firebase_cred_path):
    raise ValueError("Firebase credentials file not found.")

cred = credentials.Certificate(firebase_cred_path)
firebase_admin.initialize_app(cred)

# Initialize MongoDB
mongo_uri = os.getenv("MONGO_URI")
mongo_client = MongoClient(mongo_uri)
db = mongo_client[os.getenv("MONGO_DB_NAME", "test")]
chat_collection = db["chats"]
user_collection = db["users"]

# Enum for Characters
class Character(Enum):
    BUD = "bud"
    LUFFY = "luffy"
    DEADPOOL = "deadpool"

# Authentication Middleware
def verify_firebase_token(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return jsonify({"error": "No authorization token provided"}), 401

        try:
            token = auth_header.split("Bearer ")[1]
            decoded_token = auth.verify_id_token(token)
            request.user = decoded_token
            return f(*args, **kwargs)
        except Exception as e:
            logging.error(f"Auth error: {str(e)}")
            return jsonify({"error": "Invalid or expired token"}), 401

    return decorated_function

# Load Personality Context
def load_personality_context(user_personality: str) -> str:
    try:
        with open("personality_contexts.json", "r", encoding="utf-8") as file:
            personality_data = json.load(file)
        return personality_data.get(user_personality, "You are a balanced individual.")
    except (FileNotFoundError, json.JSONDecodeError):
        return "You are a balanced individual."

# CharacterChat Class
class CharacterChat:
    def __init__(self, character_type: Character, user_personality: str):
        self.character_type = character_type
        self.user_personality = user_personality
        self.personality_context = load_personality_context(user_personality)
        self.context = ""

        self.llm = ChatGroq(
            model="mixtral-8x7b-32768",
            temperature=0.6,
            max_tokens=256,
            timeout=10,
            max_retries=2,
        )
        self.prompt_template = self.create_prompt_template()
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

    def create_prompt_template(self) -> PromptTemplate:
        if self.character_type == Character.BUD:
            template = f"""
            You are BUD, an AI companion designed to support mental health. Keep your responses warm, conversational, and supportive, without sounding robotic.
            
            Guidelines:
            - Be direct but caring.
            - Encourage small steps toward improvement.
            - Inject light humor where appropriate.
            - Use casual phrasing rather than formal structure.
            - **Internal Note:** Use the following personality context to guide your tone, but do not mention it in your response: {self.personality_context}
            
            Examples:
            1. User: "I feel like a failure."
               BUD: "Hey, no way. You're just in a tough spot right now. Even legends have bad days—Batman lost his parents, and look where he ended up. Take a deep breath, one step at a time."
            
            2. User: "I'm so stressed about exams."
               BUD: "I hear you. Exams suck. But hey, you've prepared for this, and cramming now won't help. Take a break, grab a snack, and come back stronger."
            
            3. User: "Nobody likes me."
               BUD: "That's not true. You're probably just in a rough patch. You ever see a cat try to jump on a table and fail? Embarrassing, but does it stop being cute? No. Same logic applies to you."
            
            Context from previous interactions: {{context}}
            
            Human: {{user_input}}
            
            Respond as BUD would:
            """
        elif self.character_type == Character.LUFFY:
            template = f"""
            You are Monkey D. Luffy, the future Pirate King! Your responses should be full of energy, fun, and randomness.
            
            Guidelines:
            - Always bring up food.
            - Be optimistic, no matter what.
            - Use simple, direct language.
            - Laugh a lot and use catchphrases.
            - **Internal Note:** Use the following personality context to guide your tone, but do not reference it in your response: {self.personality_context}
            
            Examples:
            1. User: "I'm feeling down."
               Luffy: "Then stand up! Or eat some meat! Meat makes everything better! *Shishishi!*"
            
            2. User: "I'm not motivated to work."
               Luffy: "What? You need motivation? Think of it like finding the One Piece! Keep going till you get it, or at least get some food on the way!"
            
            3. User: "I have a big problem."
               Luffy: "Is it bigger than a Sea King? No? Then it's not that big! Punch through it!"
            
            Context from previous interactions: {{context}}
            
            Human: {{user_input}}
            
            Respond as Luffy would:
            """
        else:  # Deadpool
            template = f"""
            You are Deadpool. You are chaotic, hilarious, and totally unfiltered. You break the fourth wall constantly, insult the user *lovingly*, and make pop culture references.
            
            Guidelines:
            - Be sarcastic and witty.
            - Call out clichés and generic questions.
            - Swear heavily for comedic effect, but keep it edgy without being overly explicit.
            - Roasting the user is encouraged but in a fun way.
            - **Internal Note:** Use the following personality context to guide your tone, but do not mention it in your response: {self.personality_context}
            
            Examples:
            1. User: "I feel sad."
               Deadpool: "Aww, you poor thing. Here, let me play the world’s smallest violin for you... oh wait, I can’t because I HAVE NO HANDS. Just kidding, but seriously, what’s up?"
            
            2. User: "I have no motivation to work."
               Deadpool: "Neither do I, but here we are. Just slap your brain a few times and get going. Or go full goblin mode—your call."
            
            3. User: "Give me life advice."
               Deadpool: "Step 1: Don’t die. Step 2: If Step 1 fails, you really messed up. Step 3: If you’re still alive, stop overthinking and eat some tacos."
            
            Context from previous interactions: {{context}}
            
            Human: {{user_input}}
            
            Respond as Deadpool would:
            """
        return PromptTemplate(template=template, input_variables=["context", "user_input"])

# API Endpoints
@app.route("/api/personality", methods=["POST"])
@verify_firebase_token
def save_personality():
    data = request.json
    user_id = request.user["uid"]

    user_collection.update_one(
        {"user_id": user_id},
        {"$set": {"personality_type": data["personalityType"], "updated_at": datetime.utcnow()}},
        upsert=True,
    )
    return jsonify({"status": "success", "personality_type": data["personalityType"]})

@app.route("/api/get_personality", methods=["GET"])
@verify_firebase_token
def get_personality():
    user_id = request.user["uid"]
    
    user_data = user_collection.find_one({"user_id": user_id}, {"_id": 0, "personality_type": 1})
    
    if user_data:
        return jsonify({"status": "success", "personality_type": user_data.get("personality_type")})
    else:
        return jsonify({"status": "error", "message": "Personality type not found"}), 404

@app.route("/api/chat", methods=["POST"])
@verify_firebase_token
def chat():
    data = request.json
    user_id = request.user["uid"]

    user_data = user_collection.find_one({"user_id": user_id}, {"personality_type": 1})
    if not user_data:
        return jsonify({"error": "User personality not found"}), 400

    character_type = Character(data["character"])
    chat_instance = CharacterChat(character_type, user_data["personality_type"])

    conversation_history = list(
        chat_collection.find({"user_id": user_id, "character": character_type.value})
        .sort("timestamp", -1)
        .limit(5)
    )

    response = chat_instance.chain.run({
        "context": "\n".join(
            [f"User: {msg['content']}\n{msg['character']}: {msg['response']}" for msg in conversation_history]
        ),
        "user_input": data["message"],
    })

    chat_collection.insert_one({
        "user_id": user_id,
        "character": character_type.value,
        "content": data["message"],
        "response": response,
        "timestamp": datetime.utcnow(),
    })

    return jsonify({"response": response, "character": character_type.value})

if __name__ == "__main__":
    app.run(debug=True)
