import os
import re
import base64
import time
import logging
from typing import List, Dict, DefaultDict
from collections import defaultdict
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from openai import OpenAI, RateLimitError
from dotenv import load_dotenv

# Load OpenAI API key from environment variables
load_dotenv()
openai_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=openai_key)

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
MODEL = 'gpt-4o-2024-08-06'
MAX_RETRIES = 10  # Maximum number of retries after rate limit error
DELAY = 20  # Wait time in seconds before retrying

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def get_gmail_credentials() -> Credentials:
    """Handles Gmail authentication and returns credentials to access Gmail API."""
    logger.info("Authenticating Gmail credentials...")
    creds = None
    token_path = "token.json"
    
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

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
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
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
        results = service.users().messages().list(userId="me", maxResults=50, q="newer_than:7d").execute()
        messages = results.get("messages", [])
        if not messages:
            logger.info("No messages found.")
            return []

        emails = []
        for message in messages:
            msg = service.users().messages().get(userId="me", id=message["id"]).execute()
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
    while retries < MAX_RETRIES:
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                temperature=0,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            return completion.choices[0].message.content.strip()

        except RateLimitError as e:
            retries += 1
            retry_after = e.response.headers.get("Retry-After", DELAY)
            logger.warning(f"Rate limit reached. Retrying in {retry_after} seconds...")
            time.sleep(float(retry_after))
    
    logger.error("Max retries reached. Aborting.")
    raise Exception("Max retries reached. Please try again later.")

def extract_links(text: str) -> List[str]:
    """Extracts all URLs from a given text using a regex."""
    return re.findall(r'(https?://[^\s]+)', text)

def identify_top_topics(emails_text: str) -> List[str]:
    """Identifies the most important topics from all emails."""
    logging.info("Running initial topic identification.")
    prompt = f"Identify the top 5-10 most important topics discussed in the following emails:\n\n{emails_text} \
        Return only the topics in a list with no additional text."
    topics = openai_request(prompt).split('\n')
    logger.info(f"Identified {len(topics)} topics.")
    logger.info("Topics: ", topics)
    return [topic.strip() for topic in topics if topic.strip()]

def summarize_and_extract_links(topics: List[str], emails_text: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Summarizes the topics and extracts links related to each topic from the emails.
    Returns a dictionary with topic as the key and another dictionary containing 'summary' and 'links' as values.
    """
    logger.info("Summarizing topics and extracting links...")
    topic_data = defaultdict(lambda: {'summary': '', 'links': []})

    for i, topic in enumerate(topics):
        logger.info(f"Processing topic number {i}: {topic}")
        prompt = (f"Summarize the topic '{topic}' based on the following emails and extract all relevant links:\n\n"
                  f"{emails_text}\n\n"
                  "Return the summary first, followed by a list of links.")
        response = openai_request(prompt)

        # Separate summary and links using a simple heuristic (split by 'http')
        if "http" in response:
            summary, links_text = response.split("http", 1)
            summary = summary.strip()
            links = extract_links("http" + links_text)  # Add 'http' back since we split it off
        else:
            summary = response.strip()
            links = []

        topic_data[topic]['summary'] = summary
        topic_data[topic]['links'].extend(links)

    return topic_data

def format_output(topic_data: Dict[str, Dict[str, List[str]]]) -> str:
    """Formats the final output with topic summaries and relevant links."""
    logger.info("Formatting the final output...")
    final_output = []

    for topic, data in topic_data.items():
        summary = data['summary']
        links = data['links']
        link_text = "\n".join(links)
        final_output.append(f"Topic: {topic}\nSummary: {summary}\nLinks:\n{link_text}\n")

    return "\n".join(final_output)

def main():
    logger.info("Starting the email processing pipeline...")

    # Step 1: Retrieve all emails in a single batch
    email_texts = get_email_content()
    if not email_texts:
        logger.info("No email content available.")
        return

    # Combine all email content into one text
    combined_email_text = "\n\n".join(email_texts)

    # Step 2: Identify the top topics from all emails
    top_topics = identify_top_topics(combined_email_text)
    
    if not top_topics:
        logger.info("No topics found.")
        return

    # Step 3: Summarize the top topics and extract links
    topic_data = summarize_and_extract_links(top_topics, combined_email_text)

    # Step 4: Format the output
    final_output = format_output(topic_data)

    logger.info("Final Summary of Important Topics:")
    print(final_output)

if __name__ == "__main__":
    main()
