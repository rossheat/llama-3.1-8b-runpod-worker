""" Llama 3.1 8B handler file. """
import os
import json
from dotenv import load_dotenv
import runpod
import transformers
import torch
from huggingface_hub import login

load_dotenv()

HUGGING_FACE_ACCESS_TOKEN = os.getenv('HUGGING_FACE_ACCESS_TOKEN')

# Authenticate with Hugging Face
login(HUGGING_FACE_ACCESS_TOKEN)

# Load the model
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
try:
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

def parse_messages(messages):
    try:
        return json.loads(messages)
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format for messages")

def handler(job):
    """ Handler function that will be used to process jobs. """
    try:
        job_input = job['input']
        
        messages = parse_messages(job_input.get('messages'))
        
        if not messages:
            raise ValueError("No messages provided")
        
        outputs = pipeline(
            messages,
            max_new_tokens=4096,
        )
        
        generated_text = outputs[0]["generated_text"][-1]
        
        return {"status": "success", "generated_text": generated_text}
    
    except ValueError as ve:
        return {"status": "error", "message": str(ve)}
    except Exception as e:
        return {"status": "error", "message": f"An unexpected error occurred: {str(e)}"}

runpod.serverless.start({"handler": handler})