import os
import yaml
import time
import base64
import logging
from typing import List
import tiktoken
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from openai import OpenAI, RateLimitError
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()
openai_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=openai_key)

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

def get_gmail_credentials() -> Credentials:
    """Handles Gmail authentication and returns credentials to access Gmail API."""
    logger.info("Authenticating Gmail credentials...")
    creds = None
    token_path = "token.json"
    
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, [config['SCOPES']])

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                logger.info("Refreshing expired credentials...")
                creds.refresh(Request())
            except Exception as e:
                logger.error(f"Error refreshing credentials: {e}")
                creds = None
        if not creds:
            logger.info("Running OAuth flow for new credentials...")
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", [config['SCOPES']])
            creds = flow.run_local_server(port=0)
        
        # Save the credentials for the next run
        with open(token_path, "w") as token:
            token.write(creds.to_json())
            logger.info("Credentials saved to token.json")
    
    return creds

def get_email_content() -> List[str]:
    """Retrieves all the email subjects and bodies."""
    logger.info("Fetching email content...")
    creds = get_gmail_credentials()

    try:
        service = build("gmail", "v1", credentials=creds)
        results = service.users().messages().list(
            userId="me",
            labelIds=['INBOX'],
            maxResults=50,
            q="newer_than:7d",
            fields='messages(id,threadId)'
        ).execute()
        messages = results.get("messages", [])
        if not messages:
            logger.info("No messages found.")
            return []

        emails = []
        for message in messages:
            msg = service.users().messages().get(userId="me", id=message["id"], format="full").execute()
            subject = next((header["value"] for header in msg["payload"]["headers"] if header["name"] == "Subject"), None)
            body_part = next((part for part in msg["payload"].get("parts", []) if part["mimeType"] == "text/plain"), None)
            if body_part:
                body = base64.urlsafe_b64decode(body_part["body"].get("data", "")).decode("utf-8")
                emails.append(f"Subject: {subject}\nBody: {body}")
        
        logger.info(f"Fetched {len(emails)} emails.")
        return emails

    except HttpError as error:
        logger.error(f"An error occurred: {error}")
        return []

def openai_request(prompt: str) -> str:
    """Makes a request to OpenAI API with retry logic for rate limiting."""
    retries = 0
    delay = config['INITIAL_DELAY']

    while retries < config['MAX_RETRIES']:
        try:
            completion = client.chat.completions.create(
                model=config['MODEL'],
                temperature=0,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            return completion.choices[0].message.content.strip()

        except RateLimitError as e:
            retries += 1
            logger.warning(f"Rate limit reached. Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= config['BACKOFF_FACTOR']  # Exponential backoff
    
    logger.error("Max retries reached. Aborting.")
    raise Exception("Max retries reached. Please try again later.")

def generate_prompt(content, instruction):
    """Generates a prompt by combining content with an instruction."""
    return f"{instruction}\n{content}"

def count_tokens(text: str) -> int:
    """Returns the number of tokens in the given text."""
    tokenizer = tiktoken.encoding_for_model("gpt-4o")
    return len(tokenizer.encode(text))
