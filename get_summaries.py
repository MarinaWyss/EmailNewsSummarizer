import logging
from typing import List, Dict
from utils import get_email_content, openai_request

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def identify_top_topics(emails_text: str) -> List[str]:
    """Identifies the most important topics from all emails."""
    logging.info("Running initial topic identification.")
    prompt = f"Identify the top 5-10 most important topics discussed in the following emails:\n\n{emails_text} \
        Return only the topics in a list with no additional text."
    topics = openai_request(prompt).split('\n')
    logger.info(f"Identified {len(topics)} topics.")
    logger.info("Topics: ", str(topics))
    return [topic.strip() for topic in topics if topic.strip()]

def summarize_topics(topics: List[str], emails_text: str) -> Dict[str, str]:
    """
    Summarizes the topics without extracting links.
    Returns a dictionary with topics as the key and the summaries as values.
    """
    logger.info("Summarizing topics...")
    topic_summaries = {}

    for i, topic in enumerate(topics):
        logger.info(f"Processing topic number {i}: {topic}")
        prompt = f"Summarize the topic '{topic}' based on the following emails:\n\n{emails_text}"
        summary = openai_request(prompt)
        topic_summaries[topic] = summary.strip()

    return topic_summaries

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

    # Combine all email content into one text
    combined_email_text = "\n\n".join(email_texts)

    # Step 2: Identify the top topics from all emails
    top_topics = identify_top_topics(combined_email_text)
    
    if not top_topics:
        logger.info("No topics found.")
        return

    # Step 3: Summarize the top topics (without extracting links)
    topic_data = summarize_topics(top_topics, combined_email_text)

    # Step 4: Format the output
    final_output = format_output(topic_data)

    logger.info("Final Summary of Important Topics:")
    print(final_output)

if __name__ == "__main__":
    main()
