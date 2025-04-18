from collections import defaultdict
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document
import numpy as np
import re
from typing import List, Dict, Any, Optional
from gliner import GLiNER
from langchain.text_splitter import RecursiveCharacterTextSplitter



class EnhancedSemanticChunker(SemanticChunker):
    """Enhanced Semantic Chunker with sentence-level overlap and NER-aware chunking.

    This chunker extends the basic SemanticChunker with advanced features:
    1. Controllable sentence overlap between chunks for context continuity
    2. NER-aware chunk boundaries using GLiNER to preserve entity mentions
    3. Formatted entity metadata enrichment for better retrieval

    The chunker works by:
    - Finding semantically coherent boundaries using embeddings
    - Detecting entities to prevent splitting in the middle of important entities
    - Adding controlled overlap between chunks to maintain context
    - Enriching chunks with entity information for better retrieval
    """

    def __init__(
        self,
        embeddings: Any,
        gliner_model: GLiNER,
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_amount: int = 95,
        min_chunk_size: int = 5,
        max_chunk_size: Optional[int] = None,
        overlap_sentences: int = 1,
    ):
        """Initialize the enhanced semantic chunker.

        Args:
            embeddings: The embeddings model to use for semantic similarity calculation.
            gliner_model: A loaded GLiNER model instance for named entity recognition.
            breakpoint_threshold_type: Method to determine semantic breakpoints 
                ('percentile' or 'standard_deviation').
            breakpoint_threshold_amount: Threshold value for determining breakpoints 
                (higher = fewer chunks).
            min_chunk_size: Minimum number of sentences per chunk.
            max_chunk_size: Maximum number of sentences per chunk (not used in parent class).
            overlap_sentences: Number of sentences to include before and after each chunk for context.
        """
        # Initialize the parent SemanticChunker
        # This provides the basic semantic chunking functionality
        super().__init__(
            embeddings=embeddings,
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount,
            min_chunk_size=min_chunk_size,
        )
        # Store additional configuration parameters
        self.max_chunk_size = max_chunk_size  # Maximum chunk size (for future implementation)
        self.overlap_sentences = overlap_sentences  # How many sentences to overlap between chunks
        self.gliner_model = gliner_model  # GLiNER model for entity recognition
        # Entity types to extract with GLiNER
        # This comprehensive list covers most entities in business emails
        self.ner_labels = ["date", "location", "person", "action", "finance", "legal", "event", "product", "organization"]
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # ~250–300 tokens worth of characters
            chunk_overlap=0
        )

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex.
        
        A utility method to break text into sentences based on common sentence
        ending patterns (.!?). This is used both for chunking and overlap management.
        
        Args:
            text: Input text to split into sentences
            
        Returns:
            List of sentences extracted from the text
        """
        # Basic sentence splitter using regex; can replace with spaCy or nltk if needed
        # Splits on period, exclamation, or question mark followed by a space
        # Also strips whitespace from each sentence and filters out empty strings
        return [s.strip() for s in re.split(r'(?<=[.!?]) +', text) if s.strip()]

    def _get_ner_spans(self, text: str) -> List[Dict[str, int]]:
        """Extract named entity spans and information from text.
        
        A core method for entity-aware chunking. It extracts entity locations and data
        to prevent entities from being split across chunk boundaries.
        
        For shorter texts (<300 words), processes the entire text at once.
        For longer texts, delegates to _get_ner_spans_long method.
        
        Args:
            text: Input text to extract entities from
            
        Returns:
            Tuple containing (entity_spans, entity_objects)
        """
        chunks = self.splitter.split_text(text)
        all_spans = []
        all_entities = []

        try:
            for chunk in chunks:
                # GLiNER model processes text and returns entity information
                entities = self.gliner_model.predict_entities(chunk, self.ner_labels, threshold=0.5)
                # Extract start/end character positions for each entity (for boundary adjustment)
                spans = [(e['start'], e['end']) for e in entities if 'start' in e and 'end' in e]
                all_spans.extend(spans)
                all_entities.extend(entities)
            return all_spans, all_entities
        except Exception as e:
            # Log any errors in entity extraction but continue processing
            print(f"NER extraction error: {e}")
            return [], []

       
    def _format_gliner_entities(self, entities: list) -> str:
        """Format extracted entities into a human-readable text description.
        
        Creates a natural language description of entities found in the text,
        grouped by entity type, which can be used to enrich document content
        or metadata for improved retrieval.
        
        Args:
            entities: List of entity objects from GLiNER
            
        Returns:
            Formatted entity description string
        """
        if not entities:
            return ""
        
        # Group entities by type to create more readable descriptions
        grouped = defaultdict(list)
        for ent in entities:
            label = ent["label"].lower()
            text = ent["text"].strip()
            # Avoid duplicates within each entity type
            if text not in grouped[label]:
                grouped[label].append(text)

        # Format each entity type into a natural language phrase
        # This creates human-readable entity summaries for each type
        phrases = []
        for label, items in grouped.items():
            readable_items = ", ".join(items)
            # Format differently based on entity type for better readability
            if label == "person":
                phrases.append(f"people mentioned include {readable_items}")
            elif label == "date":
                phrases.append(f"dates mentioned include {readable_items}")
            elif label == "location":
                phrases.append(f"locations mentioned include {readable_items}")
            elif label == "finance":
                phrases.append(f"financial terms include {readable_items}")
            elif label == "organization":
                phrases.append(f"organizations mentioned include {readable_items}")
            elif label == "product":
                phrases.append(f"products or services mentioned include {readable_items}")
            elif label == "event":
                phrases.append(f"events mentioned include {readable_items}")
            elif label == "legal":
                phrases.append(f"legal terms mentioned include {readable_items}")
            elif label == "action":
                phrases.append(f"actions or verbs include {readable_items}")
            else:
                phrases.append(f"{label}s mentioned include {readable_items}")

        # Combine all phrases into a single description
        # This forms a comprehensive entity summary for the chunk
        return "This passage contains " + "; ".join(phrases) + ". "


    def _adjust_chunk_boundaries(self, text: str, chunks: List[str], spans: List[tuple]) -> List[str]:
        """Adjust chunk boundaries to prevent splitting entities.
        
        Ensures that named entities aren't split across chunks by extending
        chunk boundaries to fully include any entity that would be split.
        This is a key innovation in this chunker - preserving entity integrity.
        
        Args:
            text: The full source text
            chunks: List of initially determined chunks
            spans: List of entity spans (start, end) to preserve
            
        Returns:
            List of adjusted chunks with preserved entity boundaries
        """
        adjusted_chunks = []
        for chunk in chunks:
            # Find the position of this chunk in the original text
            start_idx = text.find(chunk)
            end_idx = start_idx + len(chunk)

            # Extend chunk boundaries to include any overlapping entity
            # This ensures no entity is split across chunk boundaries
            for ent_start, ent_end in spans:
                # If an entity overlaps with this chunk boundary
                if start_idx < ent_end and end_idx > ent_start:
                    # Extend the chunk to fully include the entity
                    start_idx = min(start_idx, ent_start)
                    end_idx = max(end_idx, ent_end)

            # Extract the adjusted chunk from the text
            adjusted_chunk = text[start_idx:end_idx].strip()
            adjusted_chunks.append(adjusted_chunk)

        return adjusted_chunks

    def create_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[Document]:
        """Create LangChain Document objects from texts with enhanced chunking.
        
        This method:
        1. Splits texts into semantic chunks
        2. Adjusts chunk boundaries to preserve entities
        3. Adds sentence overlap for context continuity
        4. Enriches metadata with entity information
        5. Returns Document objects ready for vectorization
        
        Args:
            texts: List of input texts to process
            metadatas: Optional list of metadata dictionaries for each text
            
        Returns:
            List of LangChain Document objects with enhanced content and metadata
        """
        # Initialize empty metadata if none provided
        if metadatas is None:
            metadatas = [{} for _ in texts]

        all_docs = []

        # Process each text with its corresponding metadata
        for i, (text, metadata) in enumerate(zip(texts, metadatas)):
            # Split text into sentences
            sentences = self._split_sentences(text)
            # Create mapping from sentence to its index for quick lookup
            sentence_to_idx = {s: idx for idx, s in enumerate(sentences)}
            
            # Extract named entities
            spans, _ = self._get_ner_spans(text)
            
            # Get initial semantic chunks using parent class method
            raw_chunks = self.split_text(text)
            
            # Adjust chunk boundaries to preserve entity mentions
            adjusted_chunks = self._adjust_chunk_boundaries(text, raw_chunks, spans)

            # Process each adjusted chunk
            for chunk in adjusted_chunks:
                # Split the chunk into sentences for overlap processing
                chunk_sentences = self._split_sentences(chunk)

                # Skip empty chunks
                if not chunk_sentences:
                    continue

                # Find original sentence indices for this chunk
                first_sentence = chunk_sentences[0]
                last_sentence = chunk_sentences[-1]
                start_idx = sentence_to_idx.get(first_sentence, 0)
                end_idx = sentence_to_idx.get(last_sentence, start_idx)

                # Add overlap sentences before and after
                # This creates continuity between chunks
                prefix = sentences[max(0, start_idx - self.overlap_sentences):start_idx]
                suffix = sentences[end_idx + 1:end_idx + 1 + self.overlap_sentences]

                # Combine into final chunk with overlap
                full_chunk = " ".join(prefix + chunk_sentences + suffix).strip()
                
                # Add entity information to metadata
                # This enriches the chunk with structured entity data
                _, chunk_entities = self._get_ner_spans(full_chunk)
                metadata["entities"] = self._format_gliner_entities(chunk_entities)
                
                
                # Create LangChain Document with prefix, enhanced content and metadata
                all_docs.append(Document(
                    page_content="passage: " + full_chunk,
                    metadata=metadata
                ))
        
        # --- Deduplication Step ---
        unique_docs_dict = {}
        for doc in all_docs:   
            # Use page_content as the key for uniqueness check
            if doc.page_content not in unique_docs_dict:
                unique_docs_dict[doc.page_content] = doc

        # Convert the dictionary values back to a list of unique documents
        all_docs = list(unique_docs_dict.values())
        # ------------------------

        return all_docs

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split existing LangChain Documents into smaller chunks.
        
        This is a convenience method for processing documents that are
        already in LangChain Document format.
        
        Args:
            documents: List of Documents to split
            
        Returns:
            List of split Documents with enhanced features
        """
        # Extract text and metadata from documents
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        # Delegate to create_documents method
        return self.create_documents(texts, metadatas)