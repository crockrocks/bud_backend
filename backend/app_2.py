from flask import Flask, request, jsonify
from inference import Character, CharacterChat, get_character_greeting, create_emotion_analyzer, get_user_personality
from inference import load_personality_context
from unsloth import FastLanguageModel
import torch

app = Flask(__name__)

# Load models for BUD character
def load_bud_model():
    bud_model, bud_tokenizer = FastLanguageModel.from_pretrained(
        model_name="fine_tuned_llama_samantha_bud",
        max_seq_length=4096,
        dtype=torch.bfloat16,
        load_in_4bit=True,
        device_map="auto",
    )
    FastLanguageModel.for_inference(bud_model)
    return bud_model, bud_tokenizer

bud_model, bud_tokenizer = load_bud_model()
emotion_chain = create_emotion_analyzer()

global_conversation_history = []
selected_character = Character.BUD
user_personality = "ISTJ"  # Default personality
dialogue_system = CharacterChat(
    character_type=selected_character,
    user_personality=user_personality,
    conversation_history=global_conversation_history,
    bud_model=bud_model,
    bud_tokenizer=bud_tokenizer
)

@app.route("/select_character", methods=["POST"])
def select_character():
    global selected_character, dialogue_system
    data = request.json
    character_name = data.get("character", "bud").lower()
    
    if character_name == "bud":
        selected_character = Character.BUD
    elif character_name == "luffy":
        selected_character = Character.LUFFY
    elif character_name == "deadpool":
        selected_character = Character.DEADPOOL
    else:
        return jsonify({"error": "Invalid character selection"}), 400
    
    dialogue_system = CharacterChat(
        character_type=selected_character,
        user_personality=user_personality,
        conversation_history=global_conversation_history,
        bud_model=bud_model if selected_character == Character.BUD else None,
        bud_tokenizer=bud_tokenizer if selected_character == Character.BUD else None
    )
    
    return jsonify({"message": get_character_greeting(selected_character)})

@app.route("/set_personality", methods=["POST"])
def set_personality():
    global user_personality, dialogue_system
    data = request.json
    personality_type = data.get("personality", "ISTJ").upper()
    
    user_personality = personality_type
    dialogue_system.user_personality = user_personality
    dialogue_system.personality_context = load_personality_context(user_personality)
    
    return jsonify({"message": f"Personality set to {user_personality}"})

@app.route("/chat", methods=["POST"])
def chat():
    global selected_character, dialogue_system
    data = request.json
    user_input = data.get("message", "").strip()
    
    if not user_input:
        return jsonify({"error": "Message cannot be empty"}), 400
    
    if user_input.lower() in ["bye", "goodbye", "exit", "quit"]:
        return jsonify({"response": "Goodbye! Come back soon!"})
    
    try:
        joy_score = emotion_chain.run(user_input=user_input)
        joy_score = max(0.0, min(1.0, float(joy_score.strip())))
    except Exception:
        joy_score = 0.5 
    
    if joy_score < 0.2 and selected_character != Character.BUD:
        selected_character = Character.BUD
        dialogue_system = CharacterChat(
            character_type=Character.BUD,
            user_personality=user_personality,
            conversation_history=global_conversation_history,
            bud_model=bud_model,
            bud_tokenizer=bud_tokenizer
        )
    
    response = dialogue_system.get_response(user_input)
    return jsonify({"character": selected_character.value, "response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

