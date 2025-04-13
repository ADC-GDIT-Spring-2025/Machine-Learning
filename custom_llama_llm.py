import os
import time
import requests
import json # Added for parsing streaming chunks
from typing import Any, Dict, Iterator, List, Optional
from dotenv import load_dotenv

load_dotenv()

from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    SystemMessage,
    HumanMessage,
    AIMessageChunk,
    BaseMessage,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field

# ========= Llama API Configuration ============
API_URL = "https://api.llms.afterhoursdev.com/completions"
API_KEY = os.environ.get("LLAMA_API_KEY")  # Make sure this environment variable is set.
print(f"API_KEY: {API_KEY}")
SESSION_TOKEN = ""
MODEL_NAME = "meta-llama3.3-70b"
SYSTEM_PROMPT = "You are a helpful assistant"
TEMPERATURE = 0.5
TOP_P = 0.9
MAX_GEN_LEN = 512

# ========= API Interaction Functions ============

def query_llama(prompt: str) -> str:
    """Sends a non-streaming request to the Llama API."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    if SESSION_TOKEN:
        headers["SESSION-TOKEN"] = SESSION_TOKEN

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "temperature": TEMPERATURE,
        "topP": TOP_P,
        "maxGenLen": MAX_GEN_LEN,
        "stream": False # Explicitly set stream to False
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status() # Raise an exception for bad status codes
        data = response.json()
        return data.get("generation", "").strip()
    except requests.exceptions.RequestException as e:
        print(f"Error querying Llama API (non-streaming): {e}")
        return "Error: Could not get response from API."

def stream_llama(prompt: str) -> Iterator[str]:
    """Sends a streaming request to the Llama API and yields text chunks."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream" # Common header for SSE
    }
    if SESSION_TOKEN:
        headers["SESSION-TOKEN"] = SESSION_TOKEN

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "temperature": TEMPERATURE,
        "topP": TOP_P,
        "maxGenLen": MAX_GEN_LEN,
        "stream": True # Enable streaming
    }

    try:
        with requests.post(API_URL, headers=headers, json=payload, stream=True) as response:
            response.raise_for_status()
            print(f"Streaming connection opened. Status: {response.status_code}")
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    # Check for Server-Sent Events (SSE) data prefix
                    if decoded_line.startswith('data: '):
                        json_content = decoded_line[len('data: '):].strip()
                        # Handle potential end-of-stream markers if necessary
                        if json_content == "[DONE]": # Example marker
                            break
                        try:
                            data = json.loads(json_content)
                            # --- Adjust the key based on actual API response --- 
                            text_chunk = data.get("generation") or data.get("chunk") or data.get("text", "")
                            if text_chunk:
                                yield str(text_chunk) # Ensure it's a string
                        except json.JSONDecodeError:
                            print(f"\nWarning: Could not decode JSON chunk: {json_content}")
                            # Optionally yield the raw content if it might be plain text
                            # yield json_content 
                    else:
                        # If not SSE, maybe it's just plain text chunks or JSON per line?
                        try:
                            data = json.loads(decoded_line)
                            text_chunk = data.get("generation") or data.get("chunk") or data.get("text", "")
                            if text_chunk:
                                yield str(text_chunk)
                        except json.JSONDecodeError:
                             # Assume it might be a plain text chunk if not JSON or SSE
                             # print(f"\nInfo: Received non-JSON line: {decoded_line}") # Debugging
                             yield decoded_line # Yielding raw line as fallback
                                 
            print("\nStreaming connection finished.")

    except requests.exceptions.RequestException as e:
        print(f"\nError during Llama API stream: {e}")
        # You might want to yield an error message or raise an exception
        yield "Error: Could not stream response from API."
    except Exception as e:
         print(f"\nAn unexpected error occurred during streaming: {e}")
         yield "Error: An unexpected error occurred."

# ========= Custom Chat Model ============
class CustomLlamaChatModel(BaseChatModel):
    """
    A general-purpose chatbot that wraps a custom Llama API call.
    Supports both standard generation and streaming.
    """
    model_name: str = MODEL_NAME
    system_prompt: str = SYSTEM_PROMPT
    temperature: float = TEMPERATURE
    top_p: float = TOP_P
    max_gen_len: int = MAX_GEN_LEN

    def _construct_prompt(self, messages: List[BaseMessage]) -> str:
        """
        Build a plain-text conversation prompt from the list of messages.
        This method converts the messages into a readable conversation string.
        """
        prompt = ""
        # Find the system message, respecting override if present
        final_system_prompt = self.system_prompt
        processed_messages = []
        for message in messages:
            if isinstance(message, SystemMessage):
                if message.content: # Use message content if provided
                    final_system_prompt = message.content
            else:
                processed_messages.append(message)
        
        # Prepend the final system prompt
        prompt += f"System: {final_system_prompt}\n"

        # Add remaining messages
        for message in processed_messages:
            if isinstance(message, HumanMessage):
                prompt += f"Human: {message.content}\n"
            elif isinstance(message, AIMessage):
                prompt += f"AI: {message.content}\n"
            else: # Handle other potential message types if necessary
                 prompt += f"{message.type}: {message.content}\n"
        return prompt

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt_text = self._construct_prompt(messages)
        # Use the NON-STREAMING query_llama function for _generate
        response_text = query_llama(prompt_text)
        
        
        # --- Token counting remains an estimation --- 
        # Proper token counting requires a tokenizer for the specific model
        # or relies on the API returning usage info.
        input_tokens = len(prompt_text.split()) # Very rough estimate
        output_tokens = len(response_text.split()) # Very rough estimate
        total_tokens = input_tokens + output_tokens

        message = AIMessage(
            content=response_text,
            response_metadata={
                "model_name": self.model_name,
                # Add other metadata if available
            },
            usage_metadata={
                 "input_tokens": input_tokens,
                 "output_tokens": output_tokens,
                 "total_tokens": total_tokens,
            }
        )
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        prompt_text = self._construct_prompt(messages)
        
        # Use the STREAMING stream_llama function
        accumulated_input_tokens = len(prompt_text.split()) # Rough estimate based on full prompt
        total_accumulated_output_tokens = 0 # Track total output tokens from API chunks
        
        for text_chunk in stream_llama(prompt_text): # Iterate over large chunks from API
            # Accumulate token count based on the actual API chunk size for final metadata
            chunk_output_tokens = len(text_chunk.split()) # Rough estimate for this large chunk
            total_accumulated_output_tokens += chunk_output_tokens
                
            # --- Client-side chunking simulation --- 
            # Break down the large chunk into smaller pieces (e.g., characters)
            for char in text_chunk: 
                # Create a small chunk containing just the character
                simulated_chunk = ChatGenerationChunk(
                    message=AIMessageChunk(
                        content=char,
                        # Usage metadata for these tiny simulated chunks is not meaningful
                        # regarding API usage, so we omit it or keep it minimal.
                        usage_metadata=None 
                    )
                )
                # Yield the small character chunk
                if run_manager:
                    # Callback manager sees the character-level token
                    run_manager.on_llm_new_token(char, chunk=simulated_chunk) 
                yield simulated_chunk
                # Add a small delay to simulate typing speed
                time.sleep(0.01) # Adjust this value for desired speed (e.g., 0.005 - 0.02)

        # --- Final Metadata Chunk (Sent AFTER all character chunks) --- 
        # Yield a final chunk with accumulated usage and response metadata if desired.
        # Calculate final metadata based on accumulated counts from the *actual* API chunks.
        final_usage_metadata = UsageMetadata(
            input_tokens=accumulated_input_tokens,
            output_tokens=total_accumulated_output_tokens,
            total_tokens=accumulated_input_tokens + total_accumulated_output_tokens
        )
        
        metadata_chunk = ChatGenerationChunk(
             message=AIMessageChunk(
                 content="", # No text content in the final metadata chunk
                 usage_metadata=final_usage_metadata,
                 response_metadata={ "model_name": self.model_name, "finish_reason": "stop" } # Example finish reason
             )
        )
        if run_manager:
            # Pass the final chunk info to the callback manager
            final_result = ChatResult(generations=[ChatGeneration(message=AIMessage(content="", usage_metadata=final_usage_metadata))])
            run_manager.on_llm_end(response=final_result)
            # run_manager.on_llm_new_token("", chunk=metadata_chunk) # Might not be needed if on_llm_end is sufficient
        yield metadata_chunk

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return "custom_llama_chat_model"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {
            "model_name": self.model_name,
            "system_prompt": self.system_prompt,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_gen_len": self.max_gen_len
        }

# ========= Usage Example ============
if __name__ == "__main__":
    llm = CustomLlamaChatModel()

    messages = [
        # SystemMessage(content="Be extremely concise."), # Example override
        SystemMessage(content=""), # Use default system prompt from class
        HumanMessage(content="Write a short story about a curious robot exploring a garden.")
        # HumanMessage(content="Hi there, how are you?")
    ]


    # --- Streaming --- 
    print("\n--- Generating Streaming Response ---")
    full_response = ""
    final_usage_metadata = None
    final_response_metadata = None

    for chunk in llm.stream(messages):
        if chunk.content:
            print(chunk.content, end="", flush=True) # Print content without extra characters
            full_response += chunk.content
        # Capture the final metadata when the stream ends (from the last chunk)
        if chunk.usage_metadata:
            final_usage_metadata = chunk.usage_metadata
            # print(f"\n[Chunk Usage: {chunk.usage_metadata}]") # Optional: print chunk metadata
        if chunk.response_metadata:
            final_response_metadata = chunk.response_metadata
            # print(f"\n[Chunk Response Meta: {chunk.response_metadata}]") # Optional: print chunk metadata
        

    print() # Final newline after streaming is complete

    print("\n--- Stream Complete ---")
    
    print("\nFinal answer:")
    print(llm.invoke(messages).content)
