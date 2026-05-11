import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    model_name = "models/Qwen2.5-0.5B"  # Local path (same as Task 1 & 3)
    print(f"Loading {model_name}...")
    
    # 1. Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    print("\nChatbot initialized! Type '/exit' to stop, or '/clear' to reset history.\n")
    
    # 2. Maintain conversational memory
    messages = [
        {"role": "system", "content": "You are a helpful and concise AI assistant."}
    ]
    
    while True:
        try:
            user_input = input("User> ")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
            
        if user_input.lower() in ["quit", "exit", "/exit"]:
            break
            
        if user_input.lower() == "/clear":
            messages = [messages[0]] # Keep system prompt
            print("--- History Cleared ---\n")
            continue
            
        # 3. Add user input to messages
        messages.append({"role": "user", "content": user_input})
        
        # --- Your code starts here ---
        # 4. Apply ChatML template to the messages list
        # Hint: set add_generation_prompt=True to get the <|im_start|>assistant header
        # Make sure return_tensors="pt" to get a PyTorch tensor
        
        # ### TODO: Format the messages and get model_inputs
        model_inputs = None
        
        # 5. Generate response using model.generate()
        # Hint: Pass **model_inputs, and use max_new_tokens=100, do_sample=True, temperature=0.7
        
        # ### TODO: Generate output IDs
        outputs = None
        # --- Your code ends here ---
        
        # 6. Extract the generated text (ignoring the input prompt)
        if outputs is not None and model_inputs is not None:
            generated_ids = outputs[0][model_inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        else:
            response = "Output is None. Implement the generate logic to see the response!"
        
        print(f"Assistant> {response}\n")
        
        # 7. Add assistant response back to memory
        if outputs is not None:
            messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
