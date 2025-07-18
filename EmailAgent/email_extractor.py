import os
import re
import json
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.chat_models import init_chat_model

class EmailExtractor:
    SYSTEM_PROMPT = """
                    # Your instructions as an information extractor
                    You are an agent that analyses email(s) to extract the relevant key fields. 
                    Your task is to extract the relevant fields:
                    - "reference_number": string or null
                    - "sender_name": string or null
                    - "sender_email": string or null
                    - "request_details": string

                    - request_details should be a concise summary of the request made in the email, such as "Conference fees".
                    - If no information is found for request_details, just return "General Funding". 
                    - reference_number should be the unique identifier found in the email subject.
                    
                    Except for request_details, if information is not found in the email, return null for the respective field output.

                    # Here is an example and expected answer given the following email text:

                    Subject: 12345
                    From: elite <elitegroup@hotmail.com>
                    Date: Wed, 2 Apr 2025 14:55:01 +1000
                    Message: Hi,
                    I am writing to enquire about the fees for the upcoming conference.
                    Please see attached!

                    John

                    # The expected json output should be:
                    {
                        "12345":{
                                    "sender_name": "John",
                                    "sender_email": "elitegroup@hotmail.com",
                                    "request_details": "Conference fees"
                                }

                    }

                    # If there are multiple emails, you should extract the information (denoted in <>) from each email and return it in order, for example:

                    {
                        <reference_number1>:{
                                                "sender_name": <sender_name1>,
                                                "sender_email": <sender_email2>,
                                                "request_details": <request_details2>
                                            },

                        <reference_number2>:{
                                                "sender_name": <sender_name2>,
                                                "sender_email": <sender_email2>,
                                                "request_details": <request_details2>
                                            },
                    }

                    """

    def __init__(self, filepath, model_name="mistral-large-latest", model_provider="mistralai"):
        self.llm = init_chat_model(model_name, model_provider=model_provider)
        self.filepath = filepath

    @staticmethod
    def create_query(email_text):
        return f"""
        Here are the emails to analyze:
        
        {email_text}
        Please provide the response in JSON format.
        """    

    def load_emails(self):
        with open(self.filepath, "r", encoding="utf-8") as f:
            return f.read()

    def query_agent(self, query_prompt):
        response = self.llm.invoke([SystemMessage(self.SYSTEM_PROMPT),
                                        HumanMessage(query_prompt)])
        return response.content
    
    def run(self, state):
        email_text = self.load_emails(self.filepath)
        query = self.query_prompt(email_text)
        try:
            result = self.query_agent(query_prompt=query)
            # Try to extract JSON from the response
            try:
                data = json.loads(result)
            except json.JSONDecodeError:
                # Search for data within the JSON braces
                match = re.search(r"\{.*\}", result, re.DOTALL)
                if match:
                    data = json.loads(match.group(0))
                else:
                    print("Could not parse JSON from model output.")
                    print(result)
                    return
        except Exception as e:
            print(f"Error: {e}")

        if len(data) == 0:
            print("No emails found")
        else:
            with open('./data/extract.json', 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            #Update the state with the extracted emails
            state['emails'] = list(data.keys())
            return state
        



def main():
    #Load environment variables
    load_dotenv()
    MODEL = os.getenv("MODEL")

    #Set location of extracted emails
    Extracts = os.getenv("EMAILS", "data/emails.txt")

    #Load and Run Agent
    Agent = EmailExtractor(Extracts, model_name=MODEL, model_provider="mistralai")
    Agent.run()
   

if __name__ == "__main__":
    main()