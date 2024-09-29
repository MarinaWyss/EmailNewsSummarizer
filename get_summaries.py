import logging
from typing import List, Dict
from utils import get_email_content, openai_request, count_tokens

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TOKEN_LIMIT = 128000  # 4o-mini token limit

def identify_top_topics(emails_text: str) -> List[str]:
    """Identifies the most important topics from all emails."""
    logging.info("Running initial topic identification.")
    prompt = f"Identify the top 5-10 most important topics discussed in the following emails:\n\n{emails_text} \
        Return only the topics in a list with no additional text."
    topics = openai_request(prompt).split('\n')
    logger.info(f"Identified {len(topics)} topics.")
    logger.info(f"Topics: {str(topics)}")
    return [topic.strip() for topic in topics if topic.strip()]

def summarize_topics(topics: List[str], emails_text: str) -> Dict[str, str]:
    """
    Summarizes the topics.
    Returns a dictionary with topics as the key and the summaries as values.
    """
    logger.info("Summarizing topics...")
    topic_summaries = {}

    for i, topic in enumerate(topics):
        logger.info(f"Processing topic number {i + 1}: {topic}")
        prompt = f"Summarize the topic '{topic}' based on the following emails:\n\n{emails_text}"
        summary = openai_request(prompt)
        topic_summaries[topic] = summary.strip()

    return topic_summaries

def process_in_batches(email_texts: List[str], token_limit: int = 4096) -> str:
    """
    Process the email content in smaller batches while staying under the token limit.
    Combines email content until the token limit is reached, then processes the batch.
    """
    combined_emails = []
    current_token_count = 0
    all_top_topics = []
    batch_results = []

    for email_text in email_texts:
        email_token_count = count_tokens(email_text)
        # If adding this email would exceed the token limit, process the current batch
        if current_token_count + email_token_count > token_limit:
            combined_email_text = "\n\n".join(combined_emails)
            top_topics = identify_top_topics(combined_email_text)
            all_top_topics.extend(top_topics)
            combined_emails = []
            current_token_count = 0
        
        combined_emails.append(email_text)
        current_token_count += email_token_count

    # Process any remaining emails
    if combined_emails:
        combined_email_text = "\n\n".join(combined_emails)
        top_topics = identify_top_topics(combined_email_text)
        all_top_topics.extend(top_topics)

    # After gathering all topics from batches, summarize the topics
    combined_emails = "\n\n".join(email_texts[:len(all_top_topics)])  # adjust to ensure content fits
    if all_top_topics:
        batch_results = summarize_topics(all_top_topics, combined_emails)

    return format_output(batch_results)

def format_output(topic_data: Dict[str, str]) -> str:
    """Formats the final output with topic summaries."""
    logger.info("Formatting the final output...")
    final_output = []

    for topic, summary in topic_data.items():
        final_output.append(f"Topic: {topic}\nSummary: {summary}\n")

    return "\n".join(final_output)

def main():
    logger.info("Starting the email processing pipeline...")

    # Step 1: Retrieve all emails in a single batch
    email_texts = get_email_content()
    if not email_texts:
        logger.info("No email content available.")
        return

    # Step 2: Process emails in batches while staying under the token limit
    final_output = process_in_batches(email_texts, TOKEN_LIMIT)

    # Step 3: Print the final output
    logger.info("Final Summary of Important Topics:")
    logger.info(final_output)

if __name__ == "__main__":
    main()
