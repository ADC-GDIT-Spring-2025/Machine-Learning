import pandas as pd
import email
import re
import spacy

# Load spaCy's small English model
nlp = spacy.load("en_core_web_sm")

def get_text_from_email(msg):
    """Extract the plain text content from an email message."""
    parts = []
    for part in msg.walk():
        if part.get_content_type() == 'text/plain':
            parts.append(part.get_payload())
    return ''.join(parts)

def split_email_addresses(line):
    """
    Separate multiple email addresses and strip extra whitespace.
    If there's only one address, return it as a string.
    If there are multiple, return them as a comma-separated string.
    """
    if line:
        addrs = list(map(lambda x: x.strip(), line.split(',')))
        return addrs[0] if len(addrs) == 1 else ', '.join(addrs)
    else:
        return None

def clean_text(text):
    """
    Clean and preprocess text:
      - Convert to lowercase.
      - Remove forwarded message markers, email headers in the body,
        message IDs, timestamps, URLs, and email addresses.
      - Remove unwanted special characters and normalize whitespace.
    """
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'---+ ?forwarded by.+?---+', ' ', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'---+ ?original message.+?---+', ' ', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'---+ ?forwarded message.+?---+', ' ', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'from:.*?(?=\n\n|\n\w)', ' ', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'sent:.*?(?=\n\n|\n\w)', ' ', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'to:.*?(?=\n\n|\n\w)', ' ', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'subject:.*?(?=\n\n|\n\w)', ' ', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'cc:.*?(?=\n\n|\n\w)', ' ', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'bcc:.*?(?=\n\n|\n\w)', ' ', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'message-id:.*', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'date:.*', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'content-transfer-encoding:.*', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'http[s]?://(?:[a-zA-Z0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' ', text)
    text = re.sub(r'[^a-zA-Z\s.,!?-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_text(text):
    """
    Tokenize text using spaCy and apply lemmatization.
    Only tokens that are alphabetic or punctuation are included.
    """
    if not text or pd.isna(text):
        return []
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.is_alpha or token.is_punct]
    return tokens

def preprocess_data(input_file, output_file, max_rows=None):
    """
    Preprocess the email data:
      - Read the CSV with a "message" column containing raw email text.
      - Parse each email to extract headers and plain text content.
      - Clean and tokenize the subject and content.
      - Process the email addresses in the From and To fields.
      - Drop any rows with empty cleaned content.
      - Return a DataFrame with the columns:
            To, From, X-To, X-From, content_clean, subject_clean, content_tokens, subject_tokens.
    """
    # Load the dataset and limit rows if necessary
    emails_df = pd.read_csv(input_file)
    if max_rows:
        emails_df = emails_df.head(max_rows)
    print(f"Processing {len(emails_df)} emails")
    
    # Parse the raw email messages from the 'message' column
    messages = list(map(email.message_from_string, emails_df['message']))
    
    # Build records from parsed messages
    records = []
    for msg in messages:
        record = {}
        # Extract header fields; default to empty string if missing
        record['subject'] = msg['Subject'] if msg['Subject'] is not None else ''
        record['From'] = msg['From'] if msg['From'] is not None else ''
        record['To'] = msg['To'] if msg['To'] is not None else ''
        record['X-To'] = msg['X-To'] if msg['X-To'] is not None else ''
        record['X-From'] = msg['X-From'] if msg['X-From'] is not None else ''
        # Extract plain text content
        record['content'] = get_text_from_email(msg)
        records.append(record)
    
    # Create DataFrame from the records
    result_df = pd.DataFrame(records)
    
    # Process email addresses for From and To fields
    result_df['From'] = result_df['From'].apply(split_email_addresses)
    result_df['To'] = result_df['To'].apply(split_email_addresses)
    
    # Clean and tokenize the subject and content
    result_df['subject_clean'] = result_df['subject'].apply(clean_text)
    result_df['subject_tokens'] = result_df['subject_clean'].apply(tokenize_text)
    result_df['content_clean'] = result_df['content'].apply(clean_text)
    result_df['content_tokens'] = result_df['content_clean'].apply(tokenize_text)
    
    # Select the desired columns
    final_df = result_df[['To', 'From', 'X-To', 'X-From', 
                          'content_clean', 'subject_clean', 
                          'content_tokens', 'subject_tokens']]
    
    # Drop rows with empty cleaned content
    final_df = final_df[final_df['content_clean'] != ""]
    
    # Save the processed DataFrame to CSV
    final_df.to_csv(output_file, index=False)
    print(f"Saved processed data to {output_file}")
    return final_df

if __name__ == "__main__":
    processed_df = preprocess_data("data/emails.csv", "data/filtered.csv", max_rows=1000)