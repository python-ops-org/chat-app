import ollama

prompt = "what is your name? Write essay on LLM"
role = "user"

messages = [{
    'role': role,
    'content': prompt,
}]

# Assuming the ollama library has a correct method for initiating a chat
try:
    stream = ollama.chat(model='gemma2', messages=messages, stream=True)
    
    for chunk in stream:
        print(chunk)  # Print the entire chunk to see its structure
        if isinstance(chunk, dict) and 'message' in chunk and 'content' in chunk['message']:
            print(chunk['message']['content'], end='', flush=True)
        else:
            print("No 'content' key found in chunk")
except AttributeError as e:
    print(f"AttributeError: {e}. Please check the library documentation for the correct method usage.")
except TypeError as e:
    print(f"TypeError: {e}. Please check the function arguments.")
except KeyError as e:
    print(f"KeyError: {e}. The expected key was not found in the chunk.")

