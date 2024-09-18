from utils import get_email_content, openai_request, generate_prompt

def summarize_email(email_text):
    """Summarizes a single email using OpenAI API."""
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
    if email_texts:
        print(f"Processing {len(email_texts)} emails.")
    else:
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
