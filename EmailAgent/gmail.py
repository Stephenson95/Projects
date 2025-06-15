import os
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import base64

#Load environment variables
load_dotenv()
GMAIL_TOKEN_PATH = os.getenv("GMAIL_TOKEN_PATH")

# Only Read Gmail messages
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def get_text_from_parts(parts):
    """
    Recursively extract text from the parts of a Gmail message
    """
    for part in parts:
        if part.get('mimeType') == 'text/plain' and 'data' in part.get('body', {}):
            return base64.urlsafe_b64decode(part['body']['data'].encode("ASCII")).decode("utf-8")
        elif part.get('parts'):
            text = get_text_from_parts(part['parts'])
            if text:
                return text
    return ''

def main():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    else:
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                # If there are no (valid) credentials available, let the user log in.
                flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)
            
            # Save the credentials for the next run
            with open('token.json', 'w') as token:
                token.write(creds.to_json())

    #Build Gmail service
    service = build('gmail', 'v1', credentials=creds)

    # Call the Gmail API to list messages
    results = service.users().messages().list(userId='me', q='is:unread').execute()
    messages = results.get('messages', [])
    #count = len(messages)

    with open('./data/emails.txt', 'w', encoding='utf-8') as f:
        for message in messages:
            msg = service.users().messages().get(userId='me', id=message['id'], format='full').execute()
            headers = msg['payload']['headers']
            subject = next((header['value'] for header in headers if header['name'] == 'Subject'), 'No Subject')
            sender = next((header['value'] for header in headers if header['name'] == 'From'), 'No From')
            date = next((header['value'] for header in headers if header['name'] == 'Date'), 'No Date')

            payload = msg.get('payload', {})
            if 'parts' in payload:
                body = get_text_from_parts(payload['parts'])
            elif payload.get('body', {}).get('data'):
                body = base64.urlsafe_b64decode(payload['body']['data'].encode("ASCII")).decode("utf-8")

            f.write(f'Subject: {subject}\n')
            f.write(f'From: {sender}\n')
            f.write(f'Date: {date}\n')
            f.write(f'Message: {body}\n\n')
        f.close()

if __name__ == '__main__':  
    main()