import re # being used in clean_text function
from dateutil import parser


"""

"""
def clean_text(text: str) -> str:
    """
    Clean text while preserving useful characters:
    - Removes weird/unprintable symbols
    - Keeps letters, numbers, basic punctuation: @ . , ? : ; ! _ ( ) &
    - Normalizes whitespace
    """

    # lowercase the texts 

    text.lower()
    # Remove anything not in the allowed set
    text = re.sub(r"[^a-z0-9@.,?;:!()&\/_ ]", '', text)
    
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    
    return text.strip()


"""

"""
def parse_email_date(date_tokens: List[str]) -> str:
    raw_date_str = " ".join(date_tokens)
    try:
        parsed = parser.parse(raw_date_str, fuzzy=True)
        return parsed.strftime("%m-%d-%Y")
    except Exception as e:
        print(f"Date parse error: {e}")
        return "unknown"

"""
"""
def extract_email_metadata(msg, df_idx):
    split_msg = msg.split()
    metadata = {}
    metadata["Message-ID"] = split_msg[split_msg.index("Message-ID:")+1]
    metadata["filename"] = df["file"].iloc[df_idx]
    try:
        metadata['sender'] = split_msg[split_msg.index("From:") + 1]
        recips = []
        try:
            for idx in range(split_msg.index("To:") + 1, split_msg.index("Subject:")):
                recips.append(split_msg[idx])
        except:
            for idx in range(split_msg.index("X-To:") + 1, split_msg.index("Subject:")):
                recips.append(split_msg[idx])
        metadata['recipient'] = " ".join(recips)
        metadata['date'] = parse_email_date(split_msg[split_msg.index("Date:") + 1: split_msg.index("Date:") + 7])
        metadata['subject'] = " ".join(split_msg[split_msg.index("Subject:") + 1:split_msg.index("Mime-Version:")])
        
    except Exception as e:
        print("Metadata extraction error:", e)
    return metadata, split_msg