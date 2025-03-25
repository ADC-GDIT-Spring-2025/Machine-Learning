# %%
import pandas as pd

df = pd.read_csv("data/emails.csv")

# %%
VECTOR_DB_NAME = "email_faiss_normalized_e5"

# %%
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document
import torch
from gliner import GLiNER
import numpy as np
import re
from typing import List, Dict, Any, Optional, Callable, Tuple

class EnhancedSemanticChunker(SemanticChunker):
    """Enhanced Semantic Chunker with NER and metadata enrichment.
    
    This chunker extends the SemanticChunker with:
    1. Controllable overlap percentage between chunks
    2. Named Entity Recognition using GLiNER
    3. Metadata enrichment for each chunk
    """
    
    def __init__(
        self,
        embeddings: Any,
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_amount: int = 95,
        min_chunk_size: int = 5,
        max_chunk_size: Optional[int] = None,
        overlap_percentage: float = 0.15,
        ner_model: Optional[Any] = None,
        ner_labels: Optional[List[str]] = None,
        metadata_extractor: Optional[Callable] = None,
    ):
        """Initialize the enhanced semantic chunker.
        
        Args:
            embeddings: The embeddings to use for semantic similarity.
            breakpoint_threshold_type: How to determine breakpoints ('percentile' or 'standard_deviation').
            breakpoint_threshold_amount: The threshold amount for breakpoints.
            min_chunk_size: Minimum chunk size in sentences.
            max_chunk_size: Maximum chunk size in sentences (not used in parent class).
            overlap_percentage: Percentage of overlap between chunks (0.0 to 1.0).
            ner_model: GLiNER model for named entity recognition.
            ner_labels: Labels to extract with NER model.
            metadata_extractor: Function to extract metadata from text.
        """
        super().__init__(
            embeddings=embeddings,
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount,
            min_chunk_size=min_chunk_size,
        )
        self.max_chunk_size = max_chunk_size  # Store as attribute but don't pass to parent
        self.overlap_percentage = overlap_percentage
        self.ner_model = ner_model
        self.ner_labels = ner_labels or ["date", "location", "person", "organization", "event"]
        self.metadata_extractor = metadata_extractor
        
        # Initialize NER model if not provided
        if self.ner_model is None and torch.cuda.is_available():
            self.ner_model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
            self.ner_model.to(torch.device('cuda'))
        elif self.ner_model is None:
            self.ner_model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
            self.ner_model.to(torch.device('cpu'))
    
    def _extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract entities from text using GLiNER.
        
        Handles long texts by splitting into smaller chunks that fit within 
        GLiNER's maximum context window (384 tokens).
        """
        if not self.ner_model:
            return {}
            
        try:
            # Initialize entity dictionary
            entity_dict = {}
            
            # Split long text into manageable chunks (roughly 384 tokens each)
            # Using a simple approach of ~100 words per chunk (~300-350 tokens typically)
            words = text.split()
            chunk_size = 100  # Approximate number of words per chunk
            
            # If text is short enough, process it directly
            if len(words) <= chunk_size:
                entities = self.ner_model.predict_entities(text, labels=self.ner_labels, threshold=0.5)
                
                # Group entities by type
                for entity in entities:
                    entity_type = entity.get('label', 'unknown')
                    if entity_type not in entity_dict:
                        entity_dict[entity_type] = []
                    entity_dict[entity_type].append(entity.get('text', ''))
            
            # For longer texts, process in chunks and combine results
            else:
                chunks = []
                for i in range(0, len(words), chunk_size):
                    chunk = " ".join(words[i:i+chunk_size])
                    chunks.append(chunk)
                
                # Process each chunk
                for chunk in chunks:
                    chunk_entities = self.ner_model.predict_entities(chunk, labels=self.ner_labels, threshold=0.5)
                    
                    # Add to combined results
                    for entity in chunk_entities:
                        entity_type = entity.get('label', 'unknown')
                        if entity_type not in entity_dict:
                            entity_dict[entity_type] = []
                        entity_dict[entity_type].append(entity.get('text', ''))
            
            return entity_dict
            
        except Exception as e:
            print(f"NER Error: {e}")
            return {}
    
    def create_documents(
        self, 
        texts: List[str], 
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[Document]:
        """Create documents with semantic chunking, overlap, and NER enrichment.
        
        Args:
            texts: List of texts to chunk.
            metadatas: Optional list of metadata dictionaries for each text.
            
        Returns:
            List of Document objects with enriched metadata.
        """
        if metadatas is None:
            metadatas = [{} for _ in texts]
            
        all_docs = []
        
        for i, (text, metadata) in enumerate(zip(texts, metadatas)):
            # First get semantic chunks without overlap
            chunks = self.split_text(text)
            
            # Calculate overlap size based on average chunk length
            if chunks:
                avg_chunk_size = sum(len(chunk.split()) for chunk in chunks) / len(chunks)
                overlap_size = int(avg_chunk_size * self.overlap_percentage)
            else:
                overlap_size = 0
            
            # Extract custom metadata if extractor is provided
            if self.metadata_extractor and callable(self.metadata_extractor):
                try:
                    custom_metadata = self.metadata_extractor(text)
                    # Ensure custom_metadata is a dictionary before updating
                    if isinstance(custom_metadata, dict):
                        metadata.update(custom_metadata)
                    else:
                        print(f"Warning: custom_metadata is not a dictionary. Type: {type(custom_metadata)}")
                except Exception as e:
                    print(f"Metadata extraction error: {str(e)}")
            
            # Create overlapping chunks with enriched metadata
            for j, chunk in enumerate(chunks):
                # Add overlap from previous chunk
                if j > 0 and overlap_size > 0:
                    prev_words = chunks[j-1].split()
                    if len(prev_words) > overlap_size:
                        overlap_text = " ".join(prev_words[-overlap_size:])
                        chunk = overlap_text + " " + chunk
                
                # Extract entities from this chunk
                entities = self._extract_entities(chunk)
                
                # Create enhanced metadata
                enhanced_metadata = metadata.copy()
                enhanced_metadata.update({
                    "chunk_index": j,
                    "total_chunks": len(chunks),
                    "entities": entities,
                    "has_overlap": j > 0 and overlap_size > 0
                })
                
                # Create document with enhanced content and metadata
                all_docs.append(Document(
                    page_content=chunk,
                    metadata=enhanced_metadata
                ))
                
        return all_docs
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents with semantic chunking, overlap, and NER enrichment.
        
        Args:
            documents: List of documents to split.
            
        Returns:
            List of split documents with enriched metadata.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return self.create_documents(texts, metadatas)

# %%

print(df.head())  
print(df.info()) 
print(df.columns) 
# %%
email_texts = df["message"].iloc[:1000].dropna().tolist()

# %%
df['file'][0]

# %%
import re

def clean_text(text: str) -> str:
    """
    Cleans the input text by:
    - Removing all special characters except @, ., ,, ?, :, ;, -, _, and space.
    - Replacing multiple spaces with a single space.
    - Stripping leading and trailing spaces.
    """
    # Keep letters, numbers, email punctuation (such as @, ., ,), and common punctuation for sentences.
    text = re.sub(r'[^A-Za-z0-9@.,?;:!()&\-_\s]', '', text)  # Allow basic email punctuation and sentence symbols
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text.strip()  # Remove leading/trailing spaces


# %%
from pprint import pprint
import random
msg = df['message'][random.randint(0,5000)]
pprint(msg)
pprint(clean_text(msg))

# %%
import faiss
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_experimental.text_splitter import SemanticChunker


# Use Microsoft E5 model instead of MPNet
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}  # Uses the cosine similarity
modelemb = HuggingFaceEmbeddings(
    model_name="intfloat/e5-base-v2", #sentence-transformers/all-mpnet-base-v2
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)


# %%
import torch
from gliner import GLiNER
from langchain_experimental.text_splitter import SemanticChunker
import spacy
import re

# ============ Setup Models ============

# Load GLiNER for NER
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gliner_model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
gliner_model.to(DEVICE)

# Load spaCy (if fallback needed)
nlp = spacy.load("en_core_web_sm")

# ============ Configuration ============
labels = ["date", "location", "person", "action", "finance", "legal", "event", "product", "organization"]

# ============ Helper Functions ============
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s.,!?-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_entities_gliner(text, labels):
    try:
        entities = gliner_model.predict_entities(text, labels=labels, threshold=0.5)
        return [e['text'] for e in entities]
    except Exception as e:
        print("GLiNER Error:", e)
        return []

def extract_email_metadata(msg):
    """Extract metadata from email message text.
    
    Args:
        msg (str): The email message text
        
    Returns:
        dict: A dictionary containing extracted metadata fields
    """
    try:
        # Check if input is valid
        if not isinstance(msg, str) or not msg.strip():
            return {}
            
        split_msg = msg.split()
        metadata = {}
        
        # Find sender
        try:
            from_index = split_msg.index("From:")
            if from_index + 1 < len(split_msg):
                metadata['sender'] = split_msg[from_index + 1]
            else:
                metadata['sender'] = ""
        except (ValueError, IndexError):
            metadata['sender'] = ""
        
        # Find recipients
        recips = []
        try:
            to_start = split_msg.index("To:") + 1
            subject_index = split_msg.index("Subject:")
            for idx in range(to_start, subject_index):
                if idx < len(split_msg):
                    recips.append(split_msg[idx])
        except (ValueError, IndexError):
            try:
                to_start = split_msg.index("X-To:") + 1
                subject_index = split_msg.index("Subject:")
                for idx in range(to_start, subject_index):
                    if idx < len(split_msg):
                        recips.append(split_msg[idx])
            except (ValueError, IndexError):
                pass
                
        metadata['recipient'] = " ".join(recips)
        
        # Find date
        try:
            date_index = split_msg.index("Date:") + 1
            if date_index < len(split_msg):
                metadata['date'] = " ".join(split_msg[date_index: min(date_index + 6, len(split_msg))])
            else:
                metadata['date'] = ""
        except (ValueError, IndexError):
            metadata['date'] = ""
            
        # Find subject
        try:
            subject_start = split_msg.index("Subject:") + 1
            try:
                mime_index = split_msg.index("Mime-Version:")
                metadata['subject'] = " ".join(split_msg[subject_start:mime_index])
            except (ValueError, IndexError):
                # Take next 10 words as subject if no Mime-Version found
                metadata['subject'] = " ".join(split_msg[subject_start:min(subject_start+10, len(split_msg))])
        except (ValueError, IndexError):
            metadata['subject'] = ""
        
        # Find main content start
        try:
            msg_start = split_msg.index("X-FileName:") + 3
            if msg_start < len(split_msg):
                metadata['content_start_index'] = msg_start
            else:
                metadata['content_start_index'] = 0
        except (ValueError, IndexError):
            metadata['content_start_index'] = 0
            
        return metadata
    except Exception as e:
        print(f"Metadata extraction error: {str(e)}")
        return {}

# %%
# Example usage of our new EnhancedSemanticChunker
enhanced_chunker = EnhancedSemanticChunker(
    embeddings=modelemb,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=5,  # More aggressive chunking
    min_chunk_size=3,
    overlap_percentage=0.15,  # 15% overlap between chunks
    ner_model=gliner_model,
    ner_labels=labels,
    metadata_extractor=extract_email_metadata,
)

# Test the enhanced chunker on a sample email
sample_email = msg
email_metadata = extract_email_metadata(sample_email)

if 'content_start_index' in email_metadata:
    split_msg = sample_email.split()
    msg_start = email_metadata['content_start_index']
    full_content = clean_text(" ".join(split_msg[msg_start:]))
    
    # Create document with the enhanced chunker
    documents = enhanced_chunker.create_documents(
        texts=[full_content],
        metadatas=[email_metadata]
    )
    
    print(f"Created {len(documents)} enhanced chunks")
    for i, doc in enumerate(documents[:2]):  # Show first two chunks only
        print(f"\n--- Chunk {i+1}/{len(documents)} ---")
        print(f"Content: {doc.page_content}...")
        print(f"Metadata: {doc.metadata}")
else:
    print("Error extracting email metadata")

# %%
len(email_texts)

# %%
import re

def clean_text(text: str) -> str:
    """
    Cleans the input text by:
    - Removing all special characters except @, ., ,, ?, :, ;, -, _, and space.
    - Replacing multiple spaces with a single space.
    - Stripping leading and trailing spaces.
    """
    # Keep letters, numbers, email punctuation (such as @, ., ,), and common punctuation for sentences.
    text = re.sub(r'[^A-Za-z0-9@.,?;:!()&\_\s]', '', text)  # Allow basic email punctuation and sentence symbols
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text.strip()  # Remove leading/trailing spaces

# %%
# Use our enhanced chunker to process all emails
async def process_emails_with_enhanced_chunker():
    from langchain_core.documents import Document
    from tqdm.notebook import tqdm
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    
    enhanced_docs = []
    
    # Process emails in batches using ThreadPoolExecutor
    def process_batch(batch_emails):
        batch_docs = []
        for email in batch_emails:
            try:
                metadata = extract_email_metadata(email)
                if 'content_start_index' in metadata:
                    split_msg = email.split()
                    msg_start = metadata['content_start_index']
                    full_content = clean_text(" ".join(split_msg[msg_start:]))
                    
                    # Create document with enhanced chunker
                    docs = enhanced_chunker.create_documents(
                        texts=[full_content],
                        metadatas=[metadata]
                    )
                    
                    # Add prefix to page content if needed
                    for doc in docs:
                        if not doc.page_content.startswith("passage:"):
                            doc.page_content = "passage: " + doc.page_content
                    
                    batch_docs.extend(docs)
            except Exception as e:
                print(f"Error processing email: {str(e)}")
        
        return batch_docs
    
    # Process in batches of 50 emails
    batch_size = 50
    batches = [email_texts[i:i+batch_size] for i in range(0, len(email_texts), batch_size)]
    
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=10) as executor:
        tasks = []
        for batch in batches:
            tasks.append(loop.run_in_executor(executor, process_batch, batch))
            
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            batch_result = await future
            enhanced_docs.extend(batch_result)
    
    return enhanced_docs

# Replace the existing processing with our enhanced version
# We need to run this in an async context, which Jupyter supports
# For non-Jupyter environments, you'd wrap this in an async function and use asyncio.run()
import asyncio
# Create a function to execute the async code
def run_async_process():
    """Run the async processing of emails and return the documents"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(process_emails_with_enhanced_chunker())

# Execute the processing
enhanced_docslist = run_async_process()

# %%
db = FAISS.from_documents(enhanced_docslist[:1], modelemb)

# %%
import numpy as np

def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """
    L2-normalizes an array.
    If the input is 1D, normalize the whole vector.
    If it's 2D, normalize each row.
    """
    if embeddings.ndim == 1:
        norm = np.linalg.norm(embeddings)
        return embeddings / norm if norm != 0 else embeddings
    else:
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norm[norm == 0] = 1  # avoid division by zero
        return embeddings / norm

# %%
len(email_texts)

# %%
len(enhanced_docslist)

# %%
from langchain_community.vectorstores import FAISS
from tqdm.asyncio import tqdm_asyncio  # Better tqdm for async

async def batch_insert(db, docslist, batch_size=40):
    tasks = []
    
    for i in range(0, len(docslist), batch_size):
        batch_docs = docslist[i : i + batch_size]
        batch_id = i // batch_size

        async def insert_batch(batch, batch_id=batch_id):
            await db.aadd_documents(batch)
            print(f"Batch {batch_id} added ({len(batch)} docs).")
        
        tasks.append(insert_batch(batch_docs))

    # Run all batch insertions concurrently with tqdm
    for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
        await coro

# Execute batch insert asynchronously
def run_batch_insert():
    """Run the batch insert asynchronously and return when complete"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(batch_insert(db, enhanced_docslist, batch_size=40))

# Execute the batch insert
run_batch_insert()

# %%
db.save_local(VECTOR_DB_NAME + "_enhanced")
print("Enhanced FAISS index updated and saved")

# %%
db = FAISS.load_local(VECTOR_DB_NAME + "_enhanced", modelemb, allow_dangerous_deserialization=True)

# %%
model = modelemb

# %%
from initialize_groq import init_groq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
        """
            Answer question only provided the context. Give a detailed answer IN minimum 5 sentences!
            SAY I DONT KNOW IF CONTEXT IS NOT ENOUGH. DONT MAKE UP ANSWERS. BUT YOU ARE FREE TO INFER/SUGGEST.
            {context}

            Here is question:
            {input}
        """
)

retriever = db.as_retriever(search_kwargs={'k':20, 'search_type':'mmr','lambda_mult':0.2})

_, llm = init_groq(model_name="llama-3.3-70b-versatile")
import random
from initialize_groq import api_keys
llm.groq_api_key = random.choice(api_keys)
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Retrieve Top-K Similar Documents (Initial Broad Search)
retriever_topk = db.as_retriever(search_kwargs={'k': 20,'fetch_k' : 100, 'search_type': 'similarity'})  # Retrieve more docs first

# MMR for Diversity (Reduce Redundant Docs)
retriever_mmr = db.as_retriever(search_kwargs={'k': 20, 'fetch_k' : 100, 'search_type': 'mmr'})  

# Create the Hybrid Retrieval Pipeline
retrieval_chain_topk = create_retrieval_chain(retriever_topk, document_chain)  # Initial broad search
retrieval_chain_mmr = create_retrieval_chain(retriever_mmr, document_chain)    # Apply MMR re-ranking

# %%
for d in enhanced_docslist:
    print(d)

# %%
import pprint
query = "give me emails related to price fixing"
pprint.pprint(retrieval_chain_topk.invoke({"input":query}))
llm.groq_api_key = random.choice(api_keys)
pprint.pprint(retrieval_chain_mmr.invoke({"input":query}))

# %%
query = "query: emails related to price fixing"
query_embedding = np.array(model.embed_query(query))
query_embedding = l2_normalize(query_embedding)  # Now a 1D vector normalized correctly

# Perform MMR search using the correctly shaped query embedding
mmr_scores = db.max_marginal_relevance_search_with_score_by_vector(
    embedding=query_embedding.tolist(),  # Pass as a list of floats
    k=20, fetch_k=100,lambda_mult=0.3
    
)

# Extract and display MMR results
mmr_list = sorted([(score, doc) for doc, score in mmr_scores], reverse=True)

for score, doc in mmr_list:
    print(f"Document: {doc.page_content[:10000]} {str(doc.metadata)[:10]} |MMR Score: {score}")


# %%
# for doc in docslist:
#     print(len(doc.page_content.split()))
    

# %%
from langchain_core.tools import tool

# Ensure retrieval_chain is correctly defined before calling this tool
@tool
def ragtool(query: str) -> str:
    """
    This is a retrieval-augmented generation (RAG) tool that queries a vector store 
    containing Enron emails.
    
    Parameters:
    query (str): The input query for retrieval.
    
    Returns:
    str: The retrieved answer from the vector store.
    """
    try:
        answer = retrieval_chain_mmr.invoke({"input": query})['answer']
        return f"Here is the ANSWER. \n ```{answer}```\n DO NOT USE THE TOOL REPEATEDLY. SHOW THE ANSWER TO THE USER. \n"
    except Exception as e:
        return f"Error: Failed to retrieve answer. Details: {str(e)}"


# %%
from typing import Literal
from langchain_core.runnables.history import RunnableWithMessageHistory
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain.memory import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langgraph.prebuilt import ToolNode, tools_condition



toolnode = ToolNode([ragtool])

def call_model(state: MessagesState):
    state["messages"]
    messages = state["messages"]
    #print(messages)
    llm.groq_api_key = random.choice(api_keys)
    llm_with_tool = llm.bind_tools([ragtool])
    response = llm_with_tool.invoke(messages)
    
    
    return {"messages": [response]}

from langgraph.graph import END
def router_function(state: MessagesState) -> Literal["tools", END]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

memory = MemorySaver()
workflow = StateGraph(MessagesState)    
workflow.add_node("agent", call_model)
workflow.add_node(toolnode)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    router_function,
    {
       "tools": "tools",
       END: END,
    },
)
workflow.add_edge("tools", "agent")
app = workflow.compile(checkpointer=memory)



# %%
from IPython.display import display_png
display_png(app.get_graph().draw_mermaid_png(),raw=True)

# %%
import time
while True:
    theinput = input("Enter something: ")
    if 'exit' in theinput:
        break
    inp = {"messages":[theinput]}
    
    config = {"configurable": {"thread_id": 1}}
    events = app.stream(inp, config=config, stream_mode="values")

    for event in events:
        event["messages"][-1].pretty_print()
    time.sleep(1)

# %%
# from typing import Literal, List
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langgraph.graph import StateGraph, START, END, MessagesState
# from langgraph.checkpoint.memory import MemorySaver
# from langchain.memory import ChatMessageHistory
# from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
# from langgraph.prebuilt import ToolNode
# from langchain_core.documents import Document
# from langchain_core.tools import tool
# from langchain.chat_models import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate

# # === Define Custom Tools ===


# @tool
# def filter_emails_by_keyword(emails: List[Document], keywords: List[str]) -> List[Document]:
#     """Filter emails that contain the given keywords in the content or metadata."""
#     def filter_email(email):
#         content = email.page_content.lower()
#         metadata = " ".join(str(val).lower() for val in email.metadata.values())
#         return any(keyword.lower() in content or keyword.lower() in metadata for keyword in keywords)
    
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         filtered_emails = list(filter(None, executor.map(lambda e: e if filter_email(e) else None, emails)))

#     return filtered_emails

# @tool
# def filter_emails_by_metadata(
#     emails: List[Document], sender: str = None, recipient: str = None, date: str = None
# ) -> List[Document]:
#     """Filter emails by metadata fields like sender, recipient, or date."""
#     def filter_email(email):
#         if sender and email.metadata.get("sender", "").lower() != sender.lower():
#             return None
#         if recipient and recipient.lower() not in email.metadata.get("recipient", "").lower():
#             return None
#         if date and date not in email.metadata.get("date", ""):
#             return None
#         return email
    
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         filtered_emails = list(filter(None, executor.map(filter_email, emails)))
    
#     return filtered_emails

# @tool
# def summarize_emails(emails: List[Document]) -> List[Document]:
#     """Summarize emails before adding them to FAISS."""
#     _,llm = init_groq(model_name="llama-3.3-70b-versatile")
#     prompt = ChatPromptTemplate.from_template("Summarize the following email:\n{email}")
#     summarized_docs = []
#     for email in emails:
#         chain = prompt | llm
#         summary = chain.invoke(email.page_content)
#         summarized_docs.append(Document(page_content=summary.content, metadata=email.metadata))
#     return summarized_docs

# toolnode = ToolNode([ragtool, filter_emails_by_keyword, filter_emails_by_metadata, summarize_emails])
# llm_with_tool = llm.bind_tools([ragtool, filter_emails_by_keyword, filter_emails_by_metadata, summarize_emails])

# # === Define Model Function ===
# def call_model(state: MessagesState):
#     """Modify agent behavior to apply filtering and summarization before RAG."""
#     messages = state["messages"]
#     query = messages[-1]

#     # Step 1: Apply Keyword Filtering
#     filtered_emails = filter_emails_by_keyword.invoke({"emails": docslist, "keywords": [query.content]})

#     # Step 2: Apply Metadata Filtering
#     filtered_emails = filter_emails_by_metadata.invoke(
#         {"emails": filtered_emails, "sender": "", "recipient": "", "date": ""}
#     )

#     # Step 3: Summarize Emails if Needed
#     summarized_emails = summarize_emails.invoke({"emails": filtered_emails})

#     # Step 4: Run RAG Tool on Filtered Emails
#     state["messages"].append("\n")
#     response = llm_with_tool.invoke([summarized_emails])

#     return {"messages": [response]}

# # === Define Router Function ===
# def router_function(state: MessagesState) -> Literal["tools", END]:
#     messages = state["messages"]
#     last_message = messages[-1]
#     if last_message.tool_calls:
#         return "tools"
#     return END

# # === Build LangGraph Workflow ===
# memory = MemorySaver()
# workflow = StateGraph(MessagesState)
# workflow.add_node("agent", call_model)
# workflow.add_node(toolnode)
# workflow.add_edge(START, "agent")
# workflow.add_conditional_edges(
#     "agent",
#     router_function,
#     {
#         "tools": "tools",
#         END: END,
#     },
# )
# workflow.add_edge("tools", "agent")
# app = workflow.compile(checkpointer=memory)

# while True:
#     theinput = input("Enter something: ")
#     if 'exit' in theinput:
#         break
#     inp = {"messages":[theinput]}

#     config = {"configurable": {"thread_id": 1}}
#     events = app.stream(inp, config=config, stream_mode="values")

#     for event in events:
#         event["messages"][-1].pretty_print()



