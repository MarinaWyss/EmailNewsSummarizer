import os
import base64
import tiktoken
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
openai_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=openai_key)

TOKEN_LIMIT = 100000
tokenizer = tiktoken.get_encoding("cl100k_base")
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
MODEL = 'gpt-4o-2024-08-06'

def count_tokens(text):
    """Returns the number of tokens in a given text."""
    return len(tokenizer.encode(text))

def split_email_by_tokens(email_text, max_tokens):
    """Splits a long email into smaller parts based on token count."""
    words = email_text.split()
    current_chunk, chunks = [], []
    current_length = 0

    for word in words:
        word_length = count_tokens(word + ' ')
        if current_length + word_length > max_tokens:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = word_length
        else:
            current_chunk.append(word)
            current_length += word_length

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def get_email_content():
    """Retrieves all the email subjects and bodies."""
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)

        with open("token.json", "w") as token:
            token.write(creds.to_json())

    try:
        service = build("gmail", "v1", credentials=creds)
        results = service.users().messages().list(userId="me", maxResults=50, q="newer_than:7d").execute()
        messages = results.get("messages", [])
        if not messages:
            print("No messages found.")
            return []

        emails = []
        for message in messages:
            msg = service.users().messages().get(userId="me", id=message["id"]).execute()
            subject = next((header["value"] for header in msg["payload"]["headers"] if header["name"] == "Subject"), None)
            body_part = next((part for part in msg["payload"].get("parts", []) if part["mimeType"] == "text/plain"), None)
            if body_part:
                body = base64.urlsafe_b64decode(body_part["body"].get("data", "")).decode("utf-8")
                emails.append(f"Subject: {subject}\nBody: {body}")
        return emails

    except HttpError as error:
        print(f"An error occurred: {error}")
        return []

def openai_request(prompt):
    """Makes a request to OpenAI API."""
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content.strip()

def generate_prompt(content, instruction):
    """Generates a prompt by combining content with an instruction."""
    return f"{instruction}\n{content}"

def summarize_email(email_text):
    """Summarizes a single email using OpenAI API."""
    if count_tokens(email_text) > TOKEN_LIMIT:
        chunks = split_email_by_tokens(email_text, TOKEN_LIMIT)
        summaries = [openai_request(generate_prompt(chunk, "Identify the key topics discussed in this email. Return the topics in a bulleted list along with links.")) for chunk in chunks]
        return " ".join(summaries)
    else:
        prompt = generate_prompt(email_text, "Identify the key topics discussed in this email. Return the topics in a bulleted list along with links.")
        return openai_request(prompt)

def batch_summarize_and_identify_topics(summaries, batch_size=5):
    """Processes summaries in batches to identify top topics."""
    batch_summaries = []
    for i in range(0, len(summaries), batch_size):
        batch = summaries[i:i + batch_size]
        combined_batch = "\n\n".join(batch)
        prompt = generate_prompt(combined_batch, "Identify the most important and frequent topics discussed in these emails. Return the topics in a bulleted list along with links.")
        batch_summaries.append(openai_request(prompt))
    return batch_summaries

def final_summary_of_top_topics(batch_summaries):
    """Takes summaries from each batch and identifies the final top topics."""
    combined_summaries = "\n\n".join(batch_summaries)
    prompt = generate_prompt(combined_summaries, "Provide a final bulleted list of the overall most important topics from these emails along with links.")
    return openai_request(prompt)

def main():
    email_texts = get_email_content()
    if not email_texts:
        print("No email content available.")
        return

    # Step 1: Summarize each email individually
    summaries = [summarize_email(email) for email in email_texts]

    # Step 2: Process summaries in batches to identify top topics
    batch_summaries = batch_summarize_and_identify_topics(summaries, batch_size=5)

    # Step 3: Generate a final summary from the top topics of each batch
    final_summary = final_summary_of_top_topics(batch_summaries)

    print("Final Summary of Important Topics from the Week:")
    print(final_summary)

if __name__ == "__main__":
    main()
