import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

model_kwargs = {'device': 'cpu'}
encode_kwars = {'normalize_embeddings': True}
model = HuggingFaceEmbeddings(
    model_name="intfloat/e5-base-v2",
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwars,
)



