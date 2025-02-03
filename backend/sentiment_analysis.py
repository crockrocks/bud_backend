from transformers import LlamaTokenizer, LlamaForCausalLM
import torch

MODEL_NAME = "lzw1008/Emollama-chat-7b"
tokenizer = LlamaTokenizer.from_pretrained(MODEL_NAME)
model = LlamaForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")

def get_emotion_intensity(text, emotion):
    prompt = f"""
    Human:
    Task: Assign a numerical value between 0 (least {emotion}) and 1 (most {emotion}) to represent the intensity of emotion {emotion} expressed in the text.
    Text: {text}
    Emotion: {emotion}
    Intensity Score:
    """
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    intensity_score = response.split(">>")[-1].strip()
    
    return intensity_score

test_cases = [
    ("I just won the lottery!!! 🎉🎉 I can’t believe this is happening, I’m so happy! 😍", "joy"),
    ("Today was rough. Nothing seems to be going right anymore.", "sadness"),
    ("Ugh, this is so annoying. I just don’t get why people can’t be on time.", "anger"),
    ("I heard strange noises outside my window at 3 AM... I was frozen in place, too scared to move.", "fear"),
    ("Happy Birthday shorty. Stay fine stay breezy stay wavy @daviistuart 😘", "joy")
]

for text, emotion in test_cases:
    score = get_emotion_intensity(text, emotion)
    print(f"\nText: {text}")
    print(f"Emotion Intensity ({emotion}): {score}")