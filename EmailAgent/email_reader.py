import os
import json
import requests
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_mistralai import MistralAIEmbeddings

#Load environment variables
load_dotenv()
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL")
MODEL = os.getenv("MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

#Initialise LLM and Embeddings
llm = init_chat_model("mistral-large-latest", model_provider="mistralai")
embeddings = MistralAIEmbeddings(model="mistral-embed")

#Load emails
EMAILS_FILE = os.path.join("data", "emails.txt")

def load_emails(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def query_ollama(prompt, model=MODEL):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(OLLAMA_API_URL, json=payload)
    response.raise_for_status()
    return response.json()["response"]

def build_prompt(email_text):
    return f"""
    You are an agent that analyzes emails for funding requests. 
    Given the following email(s), extract and return a JSON object with these fields:
    - "request_found": true/false
    - "sender_name": string or null
    - "sender_email": string or null
    - "request_details": string or null

    If there is no funding request, set "request_found" to false and the other fields to null.
    Here is the email content:
    ---
    {email_text}
    ---
    Return only the JSON object.
    """

def main():
    email_text = load_emails(EMAILS_FILE)
    prompt = build_prompt(email_text)
    try:
        result = query_ollama(prompt)
        # Try to extract JSON from the response
        try:
            data = json.loads(result)
        except json.JSONDecodeError:
            # Try to extract JSON substring if model adds text
            import re
            match = re.search(r"\{.*\}", result, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
            else:
                print("Could not parse JSON from model output.")
                print(result)
                return
        if not data.get("request_found"):
            print("no information found")
        else:
            print(json.dumps(data, indent=2))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()