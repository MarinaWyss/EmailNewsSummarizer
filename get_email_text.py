import logging
from utils import get_email_content

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def split_file_into_parts(input_string, file_name, max_words=10000):
    # Split the text into words
    words = input_string.split()

    # Calculate the number of parts needed
    num_parts = (len(words) // max_words) + (1 if len(words) % max_words != 0 else 0)

    # Split the text into parts and write each part to a new file
    for i in range(num_parts):
        part_words = words[i * max_words:(i + 1) * max_words]
        part_text = " ".join(part_words)

        # Write each part to a new file
        with open(f"{file_name}_{i + 1}.txt", "w") as output_file:
            output_file.write(part_text)

        logging.info(f"{file_name}_{i + 1}.txt has been created")

def main():
    logger.info("Starting the email processing pipeline...")

    # Step 1: Retrieve all emails in a single batch
    email_texts = get_email_content()
    if not email_texts:
        logger.info("No email content available.")
        return

    # Combine all email content into one text
    combined_email_text = "\n\n".join(email_texts)

    split_file_into_parts(combined_email_text, "email_text")

if __name__ == "__main__":
    main()
