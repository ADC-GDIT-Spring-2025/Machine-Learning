import re # being used in clean_text function
from dateutil import parser
from langchain_core.tools import tool
from typing import Literal
from langchain_core.runnables.history import RunnableWithMessageHistory
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain.memory import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langgraph.prebuilt import ToolNode, tools_condition
from initialize_groq import api_keys
from langgraph.graph import END
import time


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

# Define the agent's reasoning function
def call_model(state: MessagesState):
    state["messages"]
    messages = state["messages"]
    #print(messages)
    groqllm.groq_api_key = random.choice(api_keys)
    llm_with_tool = groqllm.bind_tools([ragtool])
    response = llm_with_tool.invoke(messages)
    
    
    return {"messages": [response]}

@tool
def ragtool(query: str, num_docs: int) -> str:
    """
    This is a retrieval-augmented generation (RAG) tool that queries a vector store 
    containing Enron emails.
    
    Parameters:
    query (str): The input query for retrieval.
    num_docs (int): The number of documents to retrieve.
    Returns:
    str: The retrieved answer from the vector store.
    """
    try:
        answer = retrieval_chain_topk.invoke({"input": query})['answer']
        return f"Here is the ANSWER. \n ```{answer}```\n DO NOT USE THE TOOL REPEATEDLY. SHOW THE ANSWER TO THE USER. \n"
    except Exception as e:
        return f"Error: Failed to retrieve answer. Details: {str(e)}"
    
def router_function(state: MessagesState) -> Literal["tools", END]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

def run_multi_query(main_query, query_generator=multi_query_chain, single_query_chain=retrieval_chain_topk):
    """
    Run a multi-query retrieval process to improve search results accuracy.
    
    This function:
    1. Takes a user query and generates multiple search queries using the LLM
    2. Executes each generated query against the retrieval system
    3. Combines and summarizes the results for a comprehensive answer
    
    Args:
        main_query: The original user question
        query_generator: Chain to generate multiple search queries
        single_query_chain: Chain to execute individual queries
        
    Returns:
        Dictionary with consolidated results and performance metrics
    """
    # Start timing the process
    start_time = total_start_time = time.time()
    
    # Generate multiple search queries from the main question
    result = query_generator.invoke({"question": main_query})
    
    # Extract the generated queries from the bullet point list
    sub_questions = [q.strip() for q in result.split('•') if q.strip()]
    
    # Record query generation time
    gen_time = time.time() - start_time
    start_time = time.time()
    
    # Track all retrieved documents and their sources
    all_docs = []
    all_results = []
    
    # Process each generated query
    print("Generated Questions:")
    for i, question in enumerate(sub_questions):
        print(f"{i+1}. {question}")
        # Execute the query against the retrieval system
        chain_result = single_query_chain.invoke({"input": question})
        all_results.append(chain_result)
        
        # Track the documents retrieved for this query
        if "context" in chain_result:
            all_docs.extend(chain_result["context"])
    
    # Record search execution time
    search_time = time.time() - start_time
    
    # Calculate combined result using the original query
    # This ensures the answer is based on all retrieved information
    final_answer = single_query_chain.invoke({
        "input": main_query,
        "context": all_docs[:10]  # Limit to top 10 most relevant documents
    })
    
    # Record total processing time
    total_time = time.time() - total_start_time
    
    # Return comprehensive results with timing metrics
    return {
        "main_query": main_query,
        "generated_queries": sub_questions,
        "individual_answers": all_results,
        "final_answer": final_answer["answer"],
        "all_docs": all_docs,
        "timing": {
            "query_generation": gen_time,
            "search_execution": search_time,
            "total_processing": total_time
        }
    }