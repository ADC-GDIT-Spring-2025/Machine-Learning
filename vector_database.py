import pandas as pd
import os
import ast
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq
os.environ["GROQ_API_KEY"]= "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"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

df = pd.read_csv('data/best_chunking_strategy.csv')
new_dataframe = False
test_questions = [
    "What does randy need to send a schedule of?",
    "What are some of randy's action items?"
]

"""
Creating Vector Database
"""
def create_vector_database_debug(df, save_path="faiss_index"):
    model_kwargs = {'device': 'cpu'}
    encode_kwars = {'normalize_embeddings': True}
    embeddings_model = HuggingFaceEmbeddings(
        model_name="intfloat/e5-base-v2",
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwars,
    )

    all_chunks = []
    all_metadata = []

    print("Starting to process chunks...")
    
    for index, row in df.iterrows():
        # Get the chunks array
        raw_chunks = row['ner_chunks']
        
        # Debug output for the first few rows
        if index < 2:
            print(f"Row {index}, raw_chunks type: {type(raw_chunks)}")
            print(f"Raw chunks sample: {raw_chunks[:2] if hasattr(raw_chunks, '__getitem__') else raw_chunks}")
        
        # Parse if it's a string
        if isinstance(raw_chunks, str):
            try:
                chunks_array = ast.literal_eval(raw_chunks)
                print(f"Row {index}: Successfully parsed string to list")
            except (SyntaxError, ValueError) as e:
                print(f"Row {index}: Error parsing chunks: {e}")
                continue  # Skip this row
        else:
            chunks_array = raw_chunks
        
        # Validate the chunks are not empty or just periods
        for i, chunk in enumerate(chunks_array):
            # Skip invalid chunks
            if not chunk or chunk == '.':
                if index < 2:
                    print(f"Skipping invalid chunk: '{chunk}'")
                continue
                
            if index < 2 and i < 2:
                print(f"Adding valid chunk: '{chunk}'")
                
            all_chunks.append(chunk)
            all_metadata.append({
                "source_row": index,
                "chunk_index": i
            })

    # Check what we've collected
    print(f"Total valid chunks collected: {len(all_chunks)}")
    if all_chunks:
        print(f"First few valid chunks: {all_chunks[:3]}")
    else:
        print("WARNING: No valid chunks found!")
        return None
    
    # Only proceed if we have valid chunks
    if not all_chunks:
        raise ValueError("No valid chunks to create vector database!")
        
    vector_database = FAISS.from_texts(
        texts=all_chunks,
        embedding=embeddings_model,
        metadatas=all_metadata
    )

    vector_database.save_local(save_path)
    print(f"Vector database created and saved to {save_path}")
    return vector_database

def create_vector_database(df, save_path="faiss_index"):
    model_kwargs = {'device': 'cpu'}
    encode_kwars = {'normalize_embeddings': True}
    embeddings_model = HuggingFaceEmbeddings(
        model_name="intfloat/e5-base-v2",
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwars,
    )

    all_chunks = []
    all_metadata = []

    for index, row in df.iterrows():
        raw_chunks = row['ner_chunks']
        chunks_array = ast.literal_eval(raw_chunks)

        for i, chunk in enumerate(chunks_array):
            all_chunks.append(chunk)
            all_metadata.append({
                "source_row": index,
                "chunk_index": i
            })

    vector_database = FAISS.from_texts(
        texts=all_chunks,
        embedding=embeddings_model,
        metadatas=all_metadata
    )

    vector_database.save_local(save_path)
    print(f"Vector database created and saved to {save_path}")
    return vector_database

"""
Loading Vector Database
"""
def load_vector_database(save_path="faiss_index"):
    model_kwargs = {'device': 'cpu'}
    encode_kwars = {'normalize_embeddings': True}
    embeddings_model = HuggingFaceEmbeddings(
        model_name="intfloat/e5-base-v2",
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwars,
    )

    if os.path.exists(save_path) and os.path.isdir(save_path):
        vector_database = FAISS.load_local(save_path, embeddings_model, allow_dangerous_deserialization=True)
        return vector_database
    else:
        raise FileNotFoundError(f"No vector database found at {save_path}")


"""
Creating LLM Model: Llama3 8B
"""
llama_llm = ChatGroq(model_name="llama3-8b-8192")


"""
Creating Prompt Template for LLM
"""
# SAY I DONT KNOW IF CONTEXT IS NOT ENOUGH. DONT MAKE UP ANSWERS. BUT YOU ARE FREE TO INFER/SUGGEST.
prompt = ChatPromptTemplate.from_template(
        """
            Answer question only provided the context. Give a detailed answer IN minimum 5 sentences!
            {context}

            Here is question:
            {input}
        """
)


"""
Running Queries Through Created RAG LLM
"""
def query_system(question, new_df):
    database_path = "faiss_index"
    if not os.path.exists(database_path) or not os.path.isdir(database_path) or new_df:
        print("Creating vector database for the first time")
        vector_database = create_vector_database(df, database_path)
    else:
        print("Loading existing vector database")
        vector_database = load_vector_database(database_path)


    # retriever from vector database
    retriever = vector_database.as_retriever(
        search_kwargs={'k': 5, 'search_type': 'mmr', 'lambda_mult': 0.5}
    )

    # print("API key set:", "REPLICATE_API_TOKEN" in os.environ)
    # create document and retrieval chain
    document_chain = create_stuff_documents_chain(llama_llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    print("Currently Querying")
    result = retrieval_chain.invoke({"input": question})

    # print("Result dictionary keys:", result.keys())
    # print("Full result structure:", result)

    print(f"Query: {question}")
    print(f"Answer: {result['answer']}\n")

    print("Retrieved chunks:")
    for i, doc in enumerate(result.get('context', [])):
        print(f"Chunk {i+1}: {doc.page_content[:100]}...")
    print("\n")
    return result


for question in test_questions:
    result = query_system(question, new_dataframe)

