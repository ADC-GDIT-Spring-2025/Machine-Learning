Skip to content
Navigation Menu
ADC-GDIT-Spring-2025
Machine-Learning

Type / to search
Code
Issues
Pull requests
Actions
Projects
Wiki
Security
Insights
Added python script that cleans + normalizes enron dataset and implements NLP preprocessing #1
 Closed
ayanban928 wants to merge 1 commit into ADC-GDIT-Spring-2025:data_preprocessing from ayanban928:data_preprocessing 
+172 −0 
 Conversation 0
 Commits 1
 Checks 0
 Files changed 1
 Closed
Added python script that cleans + normalizes enron dataset and implements NLP preprocessing
#1
File filter 
 
0 / 1 files viewed
 172 changes: 172 additions & 0 deletions172  
clean_emails.py
Viewed
Original file line number	Diff line number	Diff line change
@@ -0,0 +1,172 @@
import pandas as pd
import re
import spacy


# loading spaCy's small English model
nlp = spacy.load("en_core_web_sm")


def parse_emails(file_path):
   """
   Purpose: Parse a raw email file and extract email messages.
   """
   emails = []
   current_email = {}


   with open(file_path, 'r') as f:
       for line in f:
           line = line.strip()


           # checking for email headers
           if line.startswith(('Subject:', 'From:', 'To:', 'cc:', 'bcc:', 'Mime-Version:', 'Content-Type:', 'X-From:', 'X-To:', 'X-cc:', 'X-bcc:', 'X-Folder:', 'X-Origin:', 'X-FileName:')):
               key, value = line.split(':', 1)
               current_email[key.strip()] = value.strip()


           # checking for the start of the email body
           elif line == '' and 'body' not in current_email:
               current_email['body'] = ''


           # appending lines to the email body
           elif 'body' in current_email:
               current_email['body'] += line + '\n'


           # checking for the end of an email message
           if line.startswith('-----'):
               emails.append(current_email)
               current_email = {}


       # appending the last email if the file doesn't end with a separator
       if current_email:
           emails.append(current_email)


   return emails


def clean_text(text):
   """
   Purpose: Clean and preprocess text.
   """
   if pd.isna(text):
       return ""


   # Convert text to lowercase
   text = text.lower()


   # Remove email forward markers and headers
   text = re.sub(r'---+ ?forwarded by.+?---+', ' ', text, flags=re.DOTALL | re.IGNORECASE)
   text = re.sub(r'---+ ?original message.+?---+', ' ', text, flags=re.DOTALL | re.IGNORECASE)
   text = re.sub(r'---+ ?forwarded message.+?---+', ' ', text, flags=re.DOTALL | re.IGNORECASE)


   # Remove email headers in the body
   text = re.sub(r'from:.*?(?=\n\n|\n\w)', ' ', text, flags=re.DOTALL | re.IGNORECASE)
   text = re.sub(r'sent:.*?(?=\n\n|\n\w)', ' ', text, flags=re.DOTALL | re.IGNORECASE)
   text = re.sub(r'to:.*?(?=\n\n|\n\w)', ' ', text, flags=re.DOTALL | re.IGNORECASE)
   text = re.sub(r'subject:.*?(?=\n\n|\n\w)', ' ', text, flags=re.DOTALL | re.IGNORECASE)
   text = re.sub(r'cc:.*?(?=\n\n|\n\w)', ' ', text, flags=re.DOTALL | re.IGNORECASE)
   text = re.sub(r'bcc:.*?(?=\n\n|\n\w)', ' ', text, flags=re.DOTALL | re.IGNORECASE)


   # Remove message IDs and timestamps
   text = re.sub(r'message-id:.*', ' ', text, flags=re.IGNORECASE)
   text = re.sub(r'date:.*', ' ', text, flags=re.IGNORECASE)
   text = re.sub(r'content-transfer-encoding:.*', ' ', text, flags=re.IGNORECASE)


   # Remove URLs
   text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)


   # Remove email addresses
   text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' ', text)


   # Remove special characters and numbers
   text = re.sub(r'[^a-zA-Z\s.,!?-]', ' ', text)


   # Remove excess whitespace
   text = re.sub(r'\s+', ' ', text).strip()


   return text


def tokenize_text(text):
   """
   Purpose: Tokenize text into words and apply lemmatization using spaCy.
   """
   if not text or pd.isna(text):
       return []


   # processing text with spaCy
   doc = nlp(text)


   # extracting tokens and apply lemmatization
   tokens = [token.lemma_ for token in doc if token.is_alpha or token.is_punct]


   return tokens


def preprocess_data(input_path, output_path, max_rows=None):
   """
   Preprocess the Enron email dataset.
   """
   # parse the raw email file
   emails = parse_emails(input_path)


   # convert the list of emails to a DataFrame
   df = pd.DataFrame(emails)


   # handle missing data
   df['Subject'] = df['Subject'].fillna('')
   df['body'] = df['body'].fillna('')


   # limit the number of rows if max_rows is specified
   if max_rows:
       df = df.head(max_rows)


   print(f"Processing {len(df)} emails")


   # cleaning and tokenizing the subject
   df['subject_clean'] = df['Subject'].apply(clean_text)
   df['subject_tokens'] = df['subject_clean'].apply(tokenize_text)


   # cleaning and tokenizing the email body
   df['body_clean'] = df['body'].apply(clean_text)
   df['body_tokens'] = df['body_clean'].apply(tokenize_text)


   # selecting all the required columns
   df = df[['From', 'To', 'subject_clean', 'body_clean', 'subject_tokens', 'body_tokens']]


   # save the processed data
   print(f"Saving processed data to {output_path}...")
   df.to_csv(output_path, index=False)
   print(f"Saved processed data to {output_path}")
   return df


if __name__ == "__main__":
   processed_df = preprocess_data("data/emails.csv", "data/filtered.csv", max_rows=1000)
