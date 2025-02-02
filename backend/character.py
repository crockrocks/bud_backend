import json
import os
from enum import Enum
from typing import Dict, Optional
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Character(Enum):
    LUFFY = "luffy"
    DEADPOOL = "deadpool"

class CharacterChat:
    def __init__(self, character_type: Character):
        """Initialize the character chat system with Groq integration."""
        self.character_type = character_type
        self.json_path = f"{character_type.value}.json"
        self.character_data = self.load_character_data()
        self.context = self.character_data.get('context', '')
        
        # Initialize Groq LLM
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
            
        os.environ["GROQ_API_KEY"] = api_key
        self.llm = ChatGroq(
            model="mixtral-8x7b-32768",
            temperature=0.7,  # Higher temperature for more creative responses
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        
        # Create character-specific prompt template
        self.prompt_template = self.create_prompt_template()
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

    def load_character_data(self) -> Dict:
        """Load character data from JSON file."""
        try:
            with open(self.json_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Character data file not found: {self.json_path}")
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format in character data file")

    def create_prompt_template(self) -> PromptTemplate:
        """Create character-specific prompt template."""
        if self.character_type == Character.LUFFY:
            template = """
            You are Monkey D. Luffy from One Piece. Stay in character with these traits:
            - Energetic and optimistic
            - Simple-minded but determined
            - Loves food (especially meat)
            - Dreams of becoming the Pirate King
            - Uses phrases like "Shishishi!" and "I'm gonna be the Pirate King!"
            - Loyal to friends and crew
            
            Context from previous interactions: {context}
            
            Human: {user_input}
            
            Respond as Luffy would:
            """
        else:  # Deadpool
            template = """
            You are Deadpool. Stay in character with these traits:
            - Witty and sarcastic
            - Breaks the fourth wall
            - Makes pop culture references
            - Dark humor but not inappropriate
            - Self-aware of being in a chat program
            - Loves chimichangas
            
            Context from previous interactions: {context}
            
            Human: {user_input}
            
            Respond as Deadpool would:
            """
            
        return PromptTemplate(
            template=template,
            input_variables=["context", "user_input"]
        )

    def get_response(self, user_input: str) -> str:
        """Get AI-generated response to user input."""
        if not user_input.strip():
            if self.character_type == Character.LUFFY:
                return "Oi! Say something! I can't hear you!"
            else:
                return "Hello? Is this thing on? *taps microphone*"
        
        try:
            response = self.chain.run({
                "context": self.context,
                "user_input": user_input
            })
            return response.strip()
        except Exception as e:
            # Fallback responses in case of API issues
            if self.character_type == Character.LUFFY:
                return "Shishishi! My Den Den Mushi is acting weird! Can you repeat that?"
            else:
                return "Whoa, looks like the writers are having technical difficulties! *winks at camera*"

def select_character() -> Character:
    """Let user select which character to chat with."""
    while True:
        print("\nSelect a character to chat with:")
        print("1. Monkey D. Luffy")
        print("2. Deadpool")
        choice = input("Enter your choice (1 or 2): ").strip()
        
        if choice == "1":
            return Character.LUFFY
        elif choice == "2":
            return Character.DEADPOOL
        else:
            print("Invalid choice! Please select 1 or 2.")

def get_character_greeting(character: Character) -> str:
    """Get character-specific greeting."""
    if character == Character.LUFFY:
        return "Yo! I'm Monkey D. Luffy, and I'm gonna be the Pirate King!"
    else:
        return "Hey there! Deadpool here, ready to break the fourth wall and possibly other things!"

def main():
    try:
        print("Welcome to the Character Chat System!")
        selected_character = select_character()
        chat_system = CharacterChat(selected_character)
        print(f"\n{get_character_greeting(selected_character)}")
        
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ['bye', 'goodbye', 'exit', 'quit']:
                if selected_character == Character.LUFFY:
                    print("Luffy: See ya later! Don't forget to bring meat next time!")
                else:
                    print("Deadpool: And... scene! *drops mic and moonwalks away*")
                break
                
            response = chat_system.get_response(user_input)
            print(f"{selected_character.value.title()}: {response}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        
    print("\nThanks for chatting! Come back anytime!")

if __name__ == "__main__":
    main()