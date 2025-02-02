import re
import json

def parse_screenplay(filename):
    """
    Parse a screenplay file and extract Deadpool's dialogues into JSON format
    
    Args:
        filename (str): Path to the screenplay file
        
    Returns:
        list: List of dialogue entries in JSON format
    """
    dialogues = []
    current_scene = ""
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()
        
        # Find all scene headings
        scenes = re.finditer(r'\d+\s+(?:INT|EXT|INT\/EXT)\..*?\d+', content, re.DOTALL)
        
        for scene in scenes:
            scene_text = scene.group()
            scene_name = re.search(r'(?:INT|EXT|INT\/EXT)\..*?(?=\d+$)', scene_text).group().strip()
            
            # Find all Deadpool's dialogues in this scene
            dialogue_pattern = r'DEADPOOL\s*(?:\(.*?\))?\s*(.*?)(?=\n\s*[A-Z]{2,}|\n\s*\d+|\Z)'
            dialogues_in_scene = re.finditer(dialogue_pattern, scene_text, re.DOTALL)
            
            for dialogue in dialogues_in_scene:
                # Clean up the dialogue text
                dialogue_text = dialogue.group(1).strip()
                dialogue_text = re.sub(r'\s+', ' ', dialogue_text)
                
                if dialogue_text:  # Only add non-empty dialogues
                    entry = {
                        "scene": scene_name,
                        "character": "DEADPOOL",
                        "dialogue": dialogue_text
                    }
                    dialogues.append(entry)
    
    return dialogues

def save_to_json(dialogues, output_file):
    """
    Save the extracted dialogues to a JSON file
    
    Args:
        dialogues (list): List of dialogue entries
        output_file (str): Path to save the JSON file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({"dialogues": dialogues}, f, indent=
def main():
    input_file = "script.txt"
    output_file = "deadpool_dialogues.json"
    
    try:
        dialogues = parse_screenplay(input_file)
        save_to_json(dialogues, output_file)
        print(f"Successfully extracted {len(dialogues)} Deadpool dialogues to {output_file}")
    except Exception as e:
        print(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()