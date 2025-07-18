import os
import re
import json
import base64
from dotenv import load_dotenv
from email.mime.text import MIMEText
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.chat_models import init_chat_model
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


class EmailResponder:

    SCOPES = ['https://www.googleapis.com/auth/gmail.send']
    USER_ID = 'me'
    SYSTEM_PROMPT = """
                    You are an agent assistant for a funding department that drafts email replies to funding requests.
                    Your task is to draft a polite and professional email reply to the <sender_name> who requested the funding.
                    - Acknowledge receipt
                    - Mention the type of funding requested
                    - Mention the request will be reviewed
                    - Sign off as "Funding Team."
                    Input example:
                    {
                        "sender_name": "John",
                        "sender_email": "elitegroup@hotmail.com",
                        "request_details": "Conference fees"
                    }
                    Output example:
                    Dear John,
                    Thank you for your email regarding the funding request for Conference fees.
                    We have received your request and it will be reviewed by our team.
                    We will get back to you shortly with more information.
                    Best regards,
                    Funding Team
                    """

    def __init__(self, filepath, model_name="mistral-large-latest", model_provider="mistralai"):
        self.llm = init_chat_model(model_name, model_provider=model_provider)
        self.filepath = filepath


    @staticmethod
    def create_query(email_extract):
        return f"""
        Here is the extract to read and respond to:
        
        {email_extract}
        Please provide the response in JSON format.
        """

    def load_extracted_emails(self):
        with open(self.filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    

    def draft_email_with_llm(self, query):

        response = self.llm.invoke([SystemMessage(self.SYSTEM_PROMPT), 
                            HumanMessage(query)])
        
        # Try to extract the message from the LLM's output
        match = re.search(r"\{.*\}", response.content, re.DOTALL)
        if match:
            response = json.loads(match.group(0))
        return response
    
    def create_message(self, sender_email, subject, body):
        message = MIMEText(body['body'])
        message['to'] = sender_email
        message['from'] = self.USER_ID
        message['subject'] = subject
        return {'raw': base64.urlsafe_b64encode(message.as_bytes()).decode()}
    
    def get_gmail_service(self):
        creds = None
        if os.path.exists('token.json'):
            creds = Credentials.from_authorized_user_file('token.json', self.SCOPES)
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file('credentials.json', self.SCOPES)
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open('token.json', 'w') as token:
                token.write(creds.to_json())
        service = build('gmail', 'v1', credentials=creds)
        return service
    
    def send_message(self, service, message):
        try:
            sent = service.users().messages().send(userId=self.USER_ID, body=message).execute()
            print(f"Message sent (ID: {sent['id']})")
        except Exception as e:
            print(f"An error occurred: {e}")
    
    def run(self):
        service = self.get_gmail_service()
        emails = self.load_extracted_emails()
        for ref, info in emails.items():
            sender_email = info.get("sender_email")
            if sender_email:
                subject = f"Re: Your Funding Request Ref:{ref}"
                body = self.draft_email_with_llm(EmailResponder.create_query(info))
                message = self.create_message(sender_email, subject, body)
                self.send_message(service, message)
            else:
                print(f"Skipping entry {ref}: Missing sender email or request details.")


def main():
    #Load environment variables
    load_dotenv()
    MODEL = os.getenv("MODEL")
    
    #Set location of extracted emails
    Extracts = os.getenv("EMAILS_FILE", "data/extract.json")

    #Load and Run Agent
    Agent = EmailResponder(Extracts, model_name=MODEL, model_provider="mistralai")
    Agent.run()


if __name__ == "__main__":
    main()