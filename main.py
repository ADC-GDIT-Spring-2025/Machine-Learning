import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_experimental.text_splitter import SemanticChunker
import torch
from gliner import GLiNER
from langchain_experimental.text_splitter import SemanticChunker
import re

# Use Microsoft E5 model instead of MPNet
model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
encode_kwargs = {'normalize_embeddings': True}  # does l2 norm for the cos sim
modelemb = HuggingFaceEmbeddings(
    model_name="intfloat/e5-base-v2", 
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)


# ============ Setup Models ============

# Load GLiNER for NER
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gliner_model = GLiNER.from_pretrained(
    "glinermodellocal",
    local_files_only=False
)
gliner_model.config.max_len = 512
gliner_model.to(DEVICE)



# ============ Configuration ============
labels = ["date", "location", "person", "action", "finance", "legal", "event", "product", "organization"]

# ============ Helper Functions ============
from dateutil import parser

def parse_email_date(date_tokens: List[str]) -> str:
    raw_date_str = " ".join(date_tokens)
    try:
        parsed = parser.parse(raw_date_str, fuzzy=True)
        return parsed.strftime("%m-%d-%Y")
    except Exception as e:
        print(f"Date parse error: {e}")
        return "unknown"


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