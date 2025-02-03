import torch
from transformers import TextStreamer
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="fine_tuned_llama_samantha_bud",
    max_seq_length=4096,
    dtype=torch.bfloat16,
    load_in_4bit=True,
    device_map="auto",
)
FastLanguageModel.for_inference(model)
text_streamer = TextStreamer(tokenizer)

def generate_response(prompt, max_new_tokens=512, temperature=0.7):
    formatted_prompt = f"""<|system|>
You are BUD, an AI designed for mental health support.
<|user|>
{prompt}
<|assistant|>"""
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            streamer=text_streamer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            eos_token_id=tokenizer.encode("<|user|>")[0],  # Stop at next user token
        )
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    try:
        first_response = full_response.split("<|assistant|>")[1].split("<|user|>")[0].strip()
    except IndexError:
        first_response = full_response.split("<|assistant|>")[1].strip()
    
    return first_response

if __name__ == "__main__":
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ['quit', 'exit']:
            print("\nGoodbye!")
            break
        if user_input:
            response = generate_response(user_input)
            print("\nBUD:", response)