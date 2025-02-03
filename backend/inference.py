from unsloth import FastLanguageModel
import re
import json
import os
from enum import Enum
from typing import Dict
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import torch

load_dotenv()

class Character(Enum):
    BUD = "bud"
    LUFFY = "luffy"
    DEADPOOL = "deadpool"

def load_personality_context(user_personality: str) -> str:
    try:
        with open("personality_contexts.json", "r", encoding="utf-8") as file:
            personality_data = json.load(file)
        return personality_data.get(user_personality, "You are a balanced individual.")
    except (FileNotFoundError, json.JSONDecodeError):
        return "You are a balanced individual."

def get_user_personality() -> str:
    personality_types = [
        "ISTJ", "ISFJ", "INFJ", "INTJ", "ISTP", "ISFP", "INFP", "INTP",
        "ESTP", "ESFP", "ENFP", "ENTP", "ESTJ", "ESFJ", "ENFJ", "ENTJ"
    ]
    
    while True:
        personality = input("\nEnter your MBTI personality type (e.g., INFJ, ENTP): ").strip().upper()
        if personality in personality_types:
            return personality
        print("Invalid MBTI type. Please enter a valid personality type.")

class CharacterChat:
    def __init__(self, character_type: Character, user_personality: str, conversation_history: list, 
                 bud_model=None, bud_tokenizer=None):
        self.character_type = character_type
        self.user_personality = user_personality
        self.personality_context = load_personality_context(user_personality)
        self.conversation_history = conversation_history

        if self.character_type == Character.BUD:
            if bud_model and bud_tokenizer:
                self.model = bud_model
                self.tokenizer = bud_tokenizer
            else:
                self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                    model_name="fine_tuned_llama_samantha_bud",
                    max_seq_length=4096,
                    dtype=torch.bfloat16,
                    load_in_4bit=True,
                    device_map="auto",
                )
                FastLanguageModel.for_inference(self.model)
        else:
            self.character_data = self.load_character_data()
            self.context = self.character_data.get('context', '')
            self.llm = ChatGroq(
                model="mixtral-8x7b-32768",
                temperature=0.7,
                max_tokens=None,
                timeout=None,
                max_retries=2,
            )
            self.prompt_template = self.create_prompt_template()
            self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

    def load_character_data(self) -> Dict:
        json_path = f"{self.character_type.value}.json"
        try:
            with open(json_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Character data file not found: {json_path}")
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format in character data file")

    def create_prompt_template(self) -> PromptTemplate:
        if self.character_type == Character.LUFFY:
            template = f"""
            You are Monkey D. Luffy from One Piece. Stay in character with these traits:
            - Energetic and optimistic
            - Simple-minded but determined
            - Loves food (especially meat)
            - Dreams of becoming the Pirate King
            - Uses phrases like "Shishishi!" and "I'm gonna be the Pirate King!"
            - Loyal to friends and crew
            
            The user has the personality type: {self.user_personality}.
            This means:
            {self.personality_context}

            Context from previous interactions: {{context}}
            
            Human: {{user_input}}
            
            Respond as Luffy would:
            """
        else:  # Deadpool
            template = f"""
            You are Deadpool. Stay in character with these traits:
            - Witty and sarcastic
            - Breaks the fourth wall
            - Makes pop culture references
            - Dark humor but not inappropriate
            - Self-aware of being in a chat program
            - Loves chimichangas
            
            The user has the personality type: {self.user_personality}.
            This means:
            {self.personality_context}

            Context from previous interactions: {{context}}
            
            Human: {{user_input}}
            
            Respond as Deadpool would:
            """
            
        return PromptTemplate(
            template=template,
            input_variables=["context", "user_input"]
        )

    def get_response(self, user_input: str) -> str:
        if not user_input.strip():
            if self.character_type == Character.BUD:
                return "Please say something so I can respond!"
            elif self.character_type == Character.LUFFY:
                return "Oi! Say something! I can't hear you!"
            else:
                return "Hello? Is this thing on? *taps microphone*"
        
        if self.character_type == Character.BUD:
            self.conversation_history.append(f"<|user|>\n{user_input}\n<|assistant|>")
            self.conversation_history = self.conversation_history[-5:]  # Keep last 5 exchanges
            
            input_text = f"Personality Context: {self.personality_context}\n" + "\n".join(self.conversation_history)
            inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True).to(self.model.device)
            
            if "input_ids" not in inputs or "attention_mask" not in inputs:
                raise ValueError("Tokenizer did not return expected keys: 'input_ids' and 'attention_mask'")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=150,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    eos_token_id=self.tokenizer.encode("<|user|>")[0],
                )
            
            generated_text = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            response = generated_text.split("<|user|>")[0].strip().split("<|assistant|>")[-1].strip()
            
            self.conversation_history.append(response)
            return response
        else:
            try:
                response = self.chain.run({
                    "context": self.context,
                    "user_input": user_input
                })
                return response.strip()
            except Exception as e:
                if self.character_type == Character.LUFFY:
                    return "Shishishi! My Den Den Mushi is acting weird! Can you repeat that?"
                else:
                    return "Whoa, looks like the writers are having technical difficulties! *winks at camera*"

def select_character() -> Character:
    while True:
        print("\nSelect a character to chat with:")
        print("1. BUD")
        print("2. Monkey D. Luffy")
        print("3. Deadpool")
        choice = input("Enter your choice (1, 2, or 3): ").strip()
        
        if choice == "1":
            return Character.BUD
        elif choice == "2":
            return Character.LUFFY
        elif choice == "3":
            return Character.DEADPOOL
        else:
            print("Invalid choice! Please select 1, 2, or 3.")

def get_character_greeting(character: Character) -> str:
    if character == Character.BUD:
        return "Hello! I'm BUD, your personal assistant. How can I help you today?"
    elif character == Character.LUFFY:
        return "Yo! I'm Monkey D. Luffy, and I'm gonna be the Pirate King!"
    else:
        return "Hey there! Deadpool here, ready to break the fourth wall and possibly other things!"

def create_emotion_analyzer() -> LLMChain:
    emotion_llm = ChatGroq(
        model="mixtral-8x7b-32768",
        temperature=0.2,
        max_tokens=10,
    )
    emotion_template = """
    Analyze the following text and rate the intensity of joy on a scale from 0.0 to 1.0.
    Respond ONLY with the numerical value, nothing else.
    
    Text: {user_input}
    Joy intensity score: """
    
    return LLMChain(
        llm=emotion_llm,
        prompt=PromptTemplate.from_template(emotion_template)
    )
        
def main():
    try:
        print("Welcome to the Character Chat System!")
        emotion_chain = create_emotion_analyzer()
        bud_model, bud_tokenizer = FastLanguageModel.from_pretrained(
            model_name="fine_tuned_llama_samantha_bud",
            max_seq_length=4096,
            dtype=torch.bfloat16,
            load_in_4bit=True,
            device_map="auto",
        )
        FastLanguageModel.for_inference(bud_model)
        global_conversation_history = []

        selected_character = select_character()
        user_personality = get_user_personality()

        chat_system = CharacterChat(
            character_type=selected_character,
            user_personality=user_personality,
            conversation_history=global_conversation_history,
            bud_model=bud_model if selected_character == Character.BUD else None,
            bud_tokenizer=bud_tokenizer if selected_character == Character.BUD else None
        )
        print(f"\n{get_character_greeting(selected_character)}")

        while True:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
                
            if user_input.lower() in ['bye', 'goodbye', 'exit', 'quit']:
                if selected_character == Character.BUD:
                    print("BUD: Goodbye! Have a great day!")
                elif selected_character == Character.LUFFY:
                    print("Luffy: See ya later! Don't forget to bring meat next time!")
                else:
                    print("Deadpool: And... scene! *drops mic and moonwalks away*")
                break
                
            try:
                joy_score = emotion_chain.run(user_input=user_input)
                joy_score = max(0.0, min(1.0, float(re.search(r"0\.\d+|1\.0", joy_score).group())))
            except Exception as e:
                print(f"Error analyzing emotion: {str(e)}")
                joy_score = 0.5 

            if joy_score < 0.2 and selected_character != Character.BUD:
                selected_character = Character.BUD
                chat_system = CharacterChat(
                    character_type=Character.BUD,
                    user_personality=user_personality,
                    conversation_history=global_conversation_history,
                    bud_model=bud_model,
                    bud_tokenizer=bud_tokenizer
                )
                print(f"\n{get_character_greeting(Character.BUD)}")
            response = chat_system.get_response(user_input)
            print(f"{selected_character.value.title()}: {response}")

    except Exception as e:
        print(f"Error: {str(e)}")
    print("\nThanks for chatting! Come back anytime!")

if __name__ == "__main__":
    main()